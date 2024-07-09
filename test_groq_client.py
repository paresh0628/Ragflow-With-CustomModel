from groq import Groq

class Groqchat:
    def __init__(self, key, model_name):
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]

        ans = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf
            )
            ans = response.choices[0].message.content
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if self.is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except Exception as e:
            return ans + "\n**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf
            )
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:
                    continue
                ans += resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if self.is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens

    @staticmethod
    def is_english(text):
        # Implement your is_english function here
        return all(ord(c) < 128 for c in text)

# Example usage
groq_chat = Groqchat(key='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', model_name='llama3-8b-8192')
system_message = "You are a helpful assistant."
history = [{"role": "user", "content": "tell me joke in two line of dog"}]
gen_conf = {"temperature": 0.5, "top_p": 1, "max_tokens": 1024}
response, tokens = groq_chat.chat(system_message, history, gen_conf)
print(response)
