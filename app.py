import torch
import streamlit as st
from typing import Optional, List
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from modelscope import snapshot_download
from langchain.llms.base import LLM

# 向量模型下载
embedding_model_dir = snapshot_download('AI-ModelScope/bge-small-en-v1.5', cache_dir='./')

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-small-en-v1___5'

# 定义源大模型类
class Yuan2_LLM(LLM):
    """
    class for Yuan2_LLM
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creating tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
                                   '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
                                   '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creating model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        temperature: float = 0.7,  # 新增温度参数
        **kwargs: Any,
    ) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(
            inputs,
            do_sample=True,  # 增加随机性
            max_length=4096,
            temperature=temperature,  # 使用温度参数
        )
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

# 定义一个函数，用于获取llm和embeddings
@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings


def main():
    # 创建一个标题
    st.set_page_config(page_title="人话计算机八股文助手")
    st.title("人话计算机八股文助手")

    # 获取llm和embeddings
    llm, embeddings = get_models()

    # 用户问题回答
    st.subheader("问问题")
    user_question = st.text_input("请输入你的问题：")

    # Placeholder for showing "思考..."
    thinking_placeholder = st.empty()

    if st.button("解释"):
        if user_question:
            thinking_placeholder.text("思考中……请耐心等待一会")
            try:
                # 使用Yuan2.0模型来获取解释
                response = llm(user_question)
                answer = f"""你是一位经验丰富的计算机科学老师，擅长用简单易懂的例子解释复杂的概念。请针对下面的问题进行回答，并尽量使用生动的例子和通俗幽默的语言来帮助学生理解：
问题：{user_question}"""

                thinking_placeholder.empty()  # 清除占位符
                st.write("答案：")
                st.markdown(llm(answer))  # 直接使用Markdown输出
            except Exception as e:
                st.error(f"发生错误: {e}")

    # 随机“八股文”生成器
    st.subheader("随机八股文生成器")
    random_baguwen = ""

    # Placeholder for showing "思考..."
    thinking_placeholder_random = st.empty()

    def generate_random_baguwen(seed: int):
        torch.manual_seed(seed)  # 设置随机种子
        thinking_placeholder_random.text("思考中……请耐心等待一会")
        try:
            # 使用Yuan2.0模型生成随机内容
            random_baguwen_prompt = """你是一位经验丰富的计算机科学老师，擅长用简单易懂的例子解释复杂的概念。请你从你的数据库当中挑选并精细讲解一个计算机知识点（比如操作系统、计算机网络、计算机组成原理等方面），用通俗易懂的语言解释出来，内容可以参考包含以下元素：
    - 计算机科学的知识点
    - 生动的例子
    - 通俗幽默的语言
    - 具体的计算机应用实例
    - 与日常生活相关的计算机技术
    - 如何通过计算机技术解决问题
    - 简单的技术原理介绍
    请尽量使用通俗易懂的方式，使读者即使没有计算机背景也能理解。


    """

            random_baguwen = llm(random_baguwen_prompt, temperature=0.8)  # 提高温度参数
            thinking_placeholder_random.empty()  # 清除占位符
            return random_baguwen
        except Exception as e:
            st.error(f"发生错误: {e}")
            return ""

    # 初始随机八股文
    random_baguwen = ""

    # 按钮操作
    if st.button("生成随机八股文"):
        seed = torch.randint(0, 1000000, (1,)).item()  # 生成随机种子
        random_baguwen = generate_random_baguwen(seed)
        st.markdown("随机八股文:")  # 使用Markdown输出
        st.markdown(random_baguwen)
    # 添加“换一换”按钮
    if st.button("换一换"):
        seed = torch.randint(0, 1000000, (1,)).item()  # 生成随机种子
        thinking_placeholder_random.empty()  # 清除旧的占位符内容
        random_baguwen = generate_random_baguwen(seed)
        st.markdown("随机八股文:")  # 使用Markdown输出
        st.markdown(random_baguwen)


if __name__ == '__main__':
    main()