import asyncio
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from treelib.node import Node
from treelib.tree import Tree

from .config import Config
from .ir import IRHelper, QAModel
from .utils.retriever import rerank_answers


@dataclass
class NodeData:
    generated_sentences: List[str]
    retrieved_titles: List[str]
    retrieved_paras: List[str]
    is_main_branch: bool
    final_answer: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"generated_sentences={self.generated_sentences}, "
            f"retrieved_titles={self.retrieved_titles}, "
            f"is_main_branch={self.is_main_branch}, "
            f"final_answer={self.final_answer}"
            ")"
        )

    @property
    def repr(self) -> str:
        # For displaying in the tree with Tree.show()
        return repr(self)


class ToTRRetriever:
    def __init__(self, config: Config, seed: Optional[int] = None) -> None:
        self.helper = IRHelper(config, config.totr.retriever_gen_config, seed)

        self.search_method = config.totr.search_method
        self.branch_method = config.totr.branch_method
        self.num_samples = config.totr.num_samples
        self.max_depth = config.totr.max_depth
        self.similarity_threshold = config.totr.similarity_threshold

    async def _retrieve_dfs_branch_by_thought(self, question: str) -> Tree:
        context_tree = Tree()

        async def dfs(query: str, parent: Node) -> None:
            parent_data: NodeData = parent.data

            if (
                self.helper.should_stop(
                    parent_data.retrieved_titles, parent_data.generated_sentences
                )
                or context_tree.depth(parent) >= self.max_depth
            ):
                return

            generated_sentences = parent_data.generated_sentences.copy()
            retrieved_titles = parent_data.retrieved_titles.copy()
            retrieved_paras = parent_data.retrieved_paras.copy()

            # 1. Retrieve relevant paragraphs
            await self.helper.retrieve_one_step(
                query, retrieved_titles, retrieved_paras
            )

            # 2. Sample new thoughts
            is_main_branches: List[bool] = [parent_data.is_main_branch] + [False] * (
                self.num_samples - 1
            )
            # Generate asynchronously
            generation_tasks: Set[asyncio.Task[str]] = {
                asyncio.create_task(
                    self.helper.generate(
                        question,
                        generated_sentences,
                        retrieved_titles,
                        retrieved_paras,
                        is_main_branch,
                    ),
                    name="main" if is_main_branch else "other",
                )
                for is_main_branch in is_main_branches
            }
            dfs_tasks: List[asyncio.Task[None]] = []

            while generation_tasks:
                done, generation_tasks = await asyncio.wait(
                    generation_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    is_main_branch = task.get_name() == "main"
                    new_generation = task.result()
                    new_sentence = self.helper.get_first_sentence(new_generation)
                    if new_sentence is None:
                        if is_main_branch:
                            context_tree.create_node(
                                parent=parent,
                                data=NodeData(
                                    generated_sentences,
                                    retrieved_titles,
                                    retrieved_paras,
                                    is_main_branch,
                                ),
                            )
                        continue
                    new_generated_sentences = generated_sentences + [new_sentence]
                    # Extract answer if any
                    final_answer = self.helper.extract_answer(new_sentence)
                    if final_answer is not None:
                        # Create leaf node and exit
                        context_tree.create_node(
                            parent=parent,
                            data=NodeData(
                                new_generated_sentences,
                                retrieved_titles,
                                retrieved_paras,
                                is_main_branch,
                                final_answer,
                            ),
                        )
                        continue

                    # Create internal node and continue
                    new_node = context_tree.create_node(
                        parent=parent,
                        data=NodeData(
                            new_generated_sentences,
                            retrieved_titles,
                            retrieved_paras,
                            is_main_branch,
                        ),
                    )

                    # Update query
                    query = self.helper.get_next_query(
                        question, new_generated_sentences
                    )

                    # Traverse asynchronously
                    dfs_tasks.append(asyncio.create_task(dfs(query, new_node)))

            await asyncio.gather(*dfs_tasks)

        root = context_tree.create_node(data=NodeData([], [], [], True))
        await dfs(question, root)

        return context_tree

    def _rank_nodes_with_answers(self, nodes_with_answers: List[Node]) -> List[Node]:
        if len(nodes_with_answers) <= 1:
            return nodes_with_answers

        answers: List[str] = []
        retrieved_counts: List[int] = []
        for node in nodes_with_answers:
            answers.append(node.data.final_answer)
            retrieved_counts.append(len(node.data.retrieved_titles))

        node_ranks = rerank_answers(
            answers, retrieved_counts, self.similarity_threshold
        )
        ranked_nodes = [nodes_with_answers[i] for i in node_ranks]
        return ranked_nodes

    async def retrieve(
        self, question: str
    ) -> Tuple[List[str], List[str], Optional[str]]:
        if self.search_method == "dfs" and self.branch_method == "thought":
            context_tree = await self._retrieve_dfs_branch_by_thought(question)
        else:
            raise NotImplementedError()

        # 1. Rank branches
        nodes_with_answers: List[Node] = []
        main_branch_leaf: Optional[Node] = None
        # Extract branches with answers and main branch
        for node in context_tree.leaves():
            node_data: NodeData = node.data
            if node_data.final_answer is not None:
                nodes_with_answers.append(node)
            if node_data.is_main_branch:
                assert main_branch_leaf is None
                main_branch_leaf = node
        assert main_branch_leaf is not None, context_tree.show(data_property="repr")
        ranked_nodes = self._rank_nodes_with_answers(nodes_with_answers)
        if len(ranked_nodes) == 0:
            # Fall back to the main branch if no answers were generated
            ranked_nodes.append(main_branch_leaf)

        # 2. Select retrieved paragraphs
        retrieved_titles_list: List[List[str]] = []
        retrieved_paras_list: List[List[str]] = []
        for node in ranked_nodes:
            node_data = node.data
            retrieved_titles_list.append(node_data.retrieved_titles)
            retrieved_paras_list.append(node_data.retrieved_paras)
        retrieved_titles, retrieved_paras = self.helper.select_retrieved(
            retrieved_titles_list, retrieved_paras_list
        )
        final_answer: Optional[str] = ranked_nodes[0].data.final_answer

        return retrieved_titles, retrieved_paras, final_answer


class ToTR:
    def __init__(self, config: Config, seed: Optional[int] = None) -> None:
        self.retriever = ToTRRetriever(config, seed)
        self.qa = QAModel(config)

    async def retrieve(self, question: str) -> Tuple[List[str], List[str]]:
        titles, paras, _ = await self.retriever.retrieve(question)
        return titles, paras

    async def answer(self, question: str, use_retriever_answer: bool = False) -> str:
        titles, paras, answer = await self.retriever.retrieve(question)
        if use_retriever_answer and answer is not None:
            return answer
        answer = await self.qa.answer(question, titles, paras)
        return answer
