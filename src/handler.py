from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel
from pymongo import MongoClient
from requests.auth import _basic_auth_str

from assistant_graph import SupplierAssistant
from config import MAX_LEN_USER_PROMPT
from protection import ExceedingProtector, ProtectionStatus, ProtectorsAccumulator
from protection.base import BaseHandler
from runnables import createSupplierRunnablesVLLM


class SupplierOptions(BaseModel):
    llm_name: str
    psycopg_checkpointer: str


class SupplierHandler(BaseHandler):

    def __init__(self, options: SupplierOptions) -> None:
        self._Supplier_runnables = createSupplierRunnablesVLLM(
            llm_name=options.llm_name,
            headers={"Authorization": _basic_auth_str("admin", "password")},
        )
        self._checkpointer_db_uri = options.psycopg_checkpointer
        self._protector = ProtectorsAccumulator(protectors=[ExceedingProtector(max_len=MAX_LEN_USER_PROMPT)])
        self.mongodb_client = MongoClient(self._checkpointer_db_uri)
        self.checkpointer = MongoDBSaver(self.mongodb_client)
        self.assistant = SupplierAssistant(
            Supplier_runnables=self._Supplier_runnables,
            checkpointer=self.checkpointer,
        )

    async def ahandle_prompt(self, prompt: str, chat_id: str) -> str:
        protector_res = self._protector.check(prompt)
        if protector_res.status is not ProtectionStatus.ok:
            return protector_res.message

        config = {"configurable": {"thread_id": chat_id}}
        output = self.assistant.graph.invoke({"query": prompt, "user_id": chat_id}, config=config)
        answer = output["final_output"]
        value = output["final_output"]
        # image_data = output["image_data"] # TODO: FIX IMAGES
        return answer, value
