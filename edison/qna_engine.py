import traceback
from .edison_agents import EdisonAgents
from .models import QnaItem, AgentType, QnaAgentOutput, ExpandedQnaItem
from typing import List
from agents import Runner
from .errors import QnaEngineError
from .common.printer import Printer


class QnaEngine:
    DEFAULT_TOPIC_DETECTION = True

    def __init__(self, edison_agents: EdisonAgents, verbose: bool = False):
        """Initialize the QnaEngine with the provided Edison agents.

        Args:
            edison_agents (EdisonAgents): The Edison agents to be used for Q&A operations.
        """
        self.agents = edison_agents
        self.verbose = verbose

    async def generate_qna_pairs(
        self,
        query: str,
    ) -> List[QnaItem]:
        """Generate Q&A pairs based on the provided query.

        Args:
            query (str): The input query for which Q&A pairs are to be generated.

        Returns:
            List[QnaItem]: A list of Q&A pairs generated from the query.
        """
        try:
            qna_agent = self.agents.get_agent(agent_type=AgentType.QNA_AGENT)
            agent_query = f"""
                Generate insightful Question and Answer pairs based on the given query.
                Query: {query}
            """
            result: QnaAgentOutput = await Runner.run(
                qna_agent,
                input=agent_query,
            )
            qna_pairs = result.qna_pairs
            if not qna_pairs or len(qna_pairs) == 0:
                raise ValueError(
                    f"❌ Deep research failed for query='{query}'. Please try again later."
                )
            return qna_pairs
        except Exception as e:
            error_msg = (
                f"❌ QnA engine failed for query='{query}'. Please try again later."
            )
            if self.verbose:
                Printer.print_red_message(error_msg)
                traceback.print_exc()
            raise QnaEngineError(f"Unknown error during QnA engine processing: {e}.")

    async def expand_qna_pairs(
        self,
        qna_pairs: List[QnaItem],
        topic_detection: bool = DEFAULT_TOPIC_DETECTION,
    ) -> List[ExpandedQnaItem]:
        """Expand the Q&A pairs with additional information.

        Args:
            qna_pairs (List[QnaItem]): The list of Q&A pairs to be expanded.

        Returns:
            List[QnaItem]: The expanded list of Q&A pairs.
        """
        try:
            tasks_agent = self.agents.get_agent(agent_type=AgentType.TASKS_AGENT)

            if not qna_pairs:
                raise ValueError(
                    f"❌ Q&A pairs provided for expansion are undefined. Please provide valid Q&A pairs."
                )

            prompt = f"""
                Expand the following Q&A pairs with additional information.
                Q&A Pairs: {qna_pairs}
            """
            if topic_detection:
                prompt += f"""
                    Please also include any relevant topics or keywords that can help in understanding the context of the Q&A pairs.
                """

            agent_query = f"""
                {prompt}
                Q&A Pairs: {qna_pairs}
            """

            expanded_qna_pairs: List[ExpandedQnaItem] = []
            for qna_pair in qna_pairs:
                result = await Runner.run(
                    tasks_agent,
                    input=agent_query,
                )
                expanded_qna_pair = ExpandedQnaItem(
                    qna_pair=qna_pair,
                    expansion=result.final_output,
                )
                expanded_qna_pairs.append(expanded_qna_pair)
            return expanded_qna_pairs
        except Exception as e:
            error_msg = f"❌ QnA engine failed for Q&A pairs. Please try again later."
            if self.verbose:
                Printer.print_red_message(error_msg)
                traceback.print_exc()
            raise QnaEngineError(f"Unknown error during QnA engine processing: {e}.")

    async def run(
        self,
        query: str,
        topic_detection: bool = DEFAULT_TOPIC_DETECTION,
    ) -> List[ExpandedQnaItem]:
        """Run the Q&A engine with the provided query and topic detection flag.

        Args:
            query (str): The input query for which Q&A pairs are to be generated.
            topic_detection (bool): Flag indicating whether to perform topic detection.

        Returns:
            List[ExpandedQnaItem]: A list of expanded Q&A pairs generated from the query.
        """
        try:
            qna_pairs = await self.generate_qna_pairs(query)
            expanded_qna_pairs = await self.expand_qna_pairs(qna_pairs, topic_detection)
            return expanded_qna_pairs
        except Exception as e:
            error_msg = (
                f"❌ QnA engine failed for query='{query}'. Please try again later."
            )
            if self.verbose:
                Printer.print_red_message(error_msg)
                traceback.print_exc()
            raise QnaEngineError(f"Unknown error during QnA engine processing: {e}.")
