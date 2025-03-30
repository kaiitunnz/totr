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
    answer_confidence: Optional[float] = None

    def get_confidence(self) -> float:
        if self.answer_confidence is None:
            raise ValueError("answer_confidence is not set.")
        return self.answer_confidence

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"generated_sentences={self.generated_sentences}, "
            f"retrieved_titles={self.retrieved_titles}, "
            f"is_main_branch={self.is_main_branch}, "
            f"final_answer={self.final_answer}, "
            f"answer_confidence={self.answer_confidence}"
            ")"
        )

    @property
    def repr(self) -> str:
        # For displaying in the tree with Tree.show()
        return repr(self)


class ToTRRetriever:
    def __init__(self, config: Config, seed: Optional[int] = None) -> None:
        self.helper = IRHelper(config, config.totr.retriever_gen_config, seed=seed)

        self.search_method = config.totr.search_method
        self.branch_method = config.totr.branch_method
        self.num_samples = config.totr.num_samples
        self.beam_size = config.totr.beam_size
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

    async def _retrieve_beam_search(self, question: str) -> Tree:
        # context_tree = Tree()
        # root = context_tree.create_node(data=NodeData([], [], [], True))
        beam: List[Tuple[NodeData, str]] = [(NodeData([], [], [], True), question)]
        leaves: List[NodeData] = []

        while beam:
            new_candidates: List[Tuple[NodeData, str]] = []
            main_branch_leaf: Optional[Tuple[NodeData, str]] = None
            for parent_data, query in beam:
                # parent_data: NodeData = parent.data
                if (
                    self.helper.should_stop(
                        parent_data.retrieved_titles, parent_data.generated_sentences
                    )
                    or len(parent_data.generated_sentences) >= self.max_depth
                ):
                    leaves.append(parent_data)
                    continue

                generated_sentences = parent_data.generated_sentences.copy()
                retrieved_titles = parent_data.retrieved_titles.copy()
                retrieved_paras = parent_data.retrieved_paras.copy()

                # 1. Retrieve relevant paragraphs.
                await self.helper.retrieve_one_step(
                    query, retrieved_titles, retrieved_paras
                )

                # 2. Sample new thoughts.
                is_main_branches = [parent_data.is_main_branch] + [False] * (
                    self.beam_size - 1
                )
                generation_tasks = [
                    asyncio.create_task(
                        self.helper.generate(
                            question,
                            generated_sentences,
                            retrieved_titles,
                            retrieved_paras,
                            is_main,
                        )
                    )
                    for is_main in is_main_branches
                ]
                generation_results = await asyncio.gather(
                    *generation_tasks, return_exceptions=False
                )

                for idx, gen_result in enumerate(generation_results):
                    # Calculate answer confidence
                    answer_confidence = await self.helper.evaluate_answer_confidence(
                        question,
                        generated_sentences,
                        gen_result,
                        retrieved_titles,
                        retrieved_paras,
                    )

                    is_main_branch = is_main_branches[idx]
                    new_sentence = self.helper.get_first_sentence(gen_result)
                    if new_sentence is None:
                        if is_main_branch:
                            leaves.append(
                                NodeData(
                                    generated_sentences,
                                    retrieved_titles,
                                    retrieved_paras,
                                    is_main_branch,
                                    answer_confidence=answer_confidence,
                                )
                            )
                        continue

                    new_generated_sentences = generated_sentences + [new_sentence]
                    # Extract answer if any
                    final_answer = self.helper.extract_answer(new_sentence)
                    new_node = NodeData(
                        new_generated_sentences,
                        retrieved_titles,
                        retrieved_paras,
                        is_main_branch,
                        final_answer,
                        answer_confidence,
                    )
                    # Update query
                    query = self.helper.get_next_query(
                        question, new_generated_sentences
                    )
                    if is_main_branch:
                        assert main_branch_leaf is None
                        main_branch_leaf = (new_node, query)
                    else:
                        new_candidates.append((new_node, query))

            # If no new candidates were generated at this depth, stop.
            if not new_candidates:
                break

            new_candidates.sort(key=lambda x: x[0].get_confidence(), reverse=True)
            cur_beam_size = self.beam_size - len(leaves)
            if main_branch_leaf is None:
                beam = []
            else:
                beam = [main_branch_leaf]
                cur_beam_size -= 1
            beam.extend(new_candidates[:cur_beam_size])

        context_tree = Tree()
        root = context_tree.create_node(data=NodeData([], [], [], True))
        for leaf in leaves:
            context_tree.create_node(data=leaf, parent=root)

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
        elif self.search_method == "beam" and self.branch_method == "thought":
            context_tree = await self._retrieve_beam_search(question)
        else:
            raise NotImplementedError()

        # 1. Rank branches
        leaves = context_tree.leaves()
        main_branch_leaf: Optional[Node] = None
        # Extract branches with answers and main branch
        for node in leaves:
            node_data: NodeData = node.data
            if node_data.final_answer is None:
                node_data.final_answer = await self.helper.get_answer(
                    question,
                    node_data.generated_sentences,
                    node_data.retrieved_titles,
                    node_data.retrieved_paras,
                )
            if node_data.is_main_branch:
                assert main_branch_leaf is None
                main_branch_leaf = node
        assert main_branch_leaf is not None, context_tree.show(data_property="repr")
        ranked_nodes = self._rank_nodes_with_answers(leaves)
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
        self.use_retriever_answer = config.qa.use_retriever_answer

    async def retrieve(self, question: str) -> Tuple[List[str], List[str]]:
        titles, paras, _ = await self.retriever.retrieve(question)
        return titles, paras

    async def answer(
        self, question: str, use_retriever_answer: Optional[bool] = None
    ) -> str:
        titles, paras, answer = await self.retriever.retrieve(question)
        if use_retriever_answer is None:
            use_retriever_answer = self.use_retriever_answer
        if use_retriever_answer and answer is not None:
            return answer
        answer = await self.qa.answer(question, titles, paras)
        return answer
