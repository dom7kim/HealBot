from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

from langchain import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Union 
import re
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

st.markdown("<h1 style='font-size: 24px;'>🦜️🔗 HealBot: Expert Medical Advice at Your Fingertips </h1>", unsafe_allow_html=True)

search = DuckDuckGoSearchRun()

def duck_wrapper(input_text):
    search_results = search.run(f"site:healthline.com OR site:webmd.com {input_text}")
    return search_results

tools = [
    Tool(
        name = "Search Web",
        func=duck_wrapper,
        description="useful for when you need to answer medical and pharmalogical questions"
    )
]

# Set up the base template
template_with_history = """Please answer the following question to the best of your ability, speaking as a compassionate medical professional. You have access to the tools listed below:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer with the expertise of a compassionate medical professional. If the condition seems serious, recommend seeking consultation with a doctor.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    @staticmethod
    def format_intermediate_steps(intermediate_steps) -> str:
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        return thoughts

    @staticmethod
    def format_tools(tools: List[Tool]) -> Tuple[str, str]:
        tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        return tools_description, tool_names

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        kwargs["agent_scratchpad"] = self.format_intermediate_steps(intermediate_steps)
        
        tools_description, tool_names = self.format_tools(self.tools)
        kwargs["tools"] = tools_description
        kwargs["tool_names"] = tool_names
        
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    
    @staticmethod
    def parse_final_answer(llm_output: str) -> str:
        return llm_output.split("Final Answer:")[-1].strip()

    @staticmethod
    def parse_action_and_input(llm_output: str) -> Tuple[str, str]:
        regex_pattern = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex_pattern, llm_output, re.DOTALL)

        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        return action, action_input

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": self.parse_final_answer(llm_output)},
                log=llm_output,
            )

        # Parse out the action and action input
        action, action_input = self.parse_action_and_input(llm_output)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)

output_parser = CustomOutputParser()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm HealBot, here to provide expert medical advice at your fingertips. What can I help you with today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

llm = OpenAI(temperature=0, streaming=True)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                    tools=tools, 
                                                    verbose=True,
                                                    memory=memory)

if prompt := st.chat_input(placeholder="How can I treat a sprained ankle?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_executor.run(st.session_state.messages[-1]['content'], callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)