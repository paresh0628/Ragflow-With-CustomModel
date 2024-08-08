from google.generativeai import GenerativeModel, configure


class GeminiChat:
    def __init__(self, key, model_name, generation_config):
        configure(api_key=key)
        self.model_name = model_name
        self.generation_config = generation_config

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "model", "content": system})

        # Convert history to the expected format
        formatted_history = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in history]

        # Merge generation config with gen_conf
        gen_conf = {**self.generation_config, **gen_conf}

        ans = ""
        try:
            model = GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_conf
            )
            chat_session = model.start_chat(
                history=formatted_history
            )
            response = chat_session.send_message(formatted_history[-1]['parts'][0]['text'])
            ans = response.text
            return ans, None  # Omit total_tokens
        except Exception as e:
            return ans + "\n**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "model", "content": system})

        # Convert history to the expected format
        formatted_history = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in history]

        # Merge generation config with gen_conf
        gen_conf = {**self.generation_config, **gen_conf}
        ans = ""
        try:
            model = GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_conf
            )
            chat_session = model.start_chat(
                history=formatted_history
            )
            response = chat_session.send_message(formatted_history[-1]['parts'][0]['text'], stream=True)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:
                    continue
                ans += resp.choices[0].delta.content
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

    @staticmethod
    def is_english(text):
        # Implement your is_english function here
        return all(ord(c) < 128 for c in text)


# Example usage
gemini_chat = GeminiChat(
    key='AIzaSyBhzR6Ule_CqGGuT9numg6wYvQTMAglBeQ',
    model_name='gemini-1.5-pro',
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain"
    }
)
system_message = "You are a helpful assistant."
history = [{"role": "user", "content": "tell me a  joke on cats"}]
gen_conf = {"temperature": 0.5, "top_p": 1, "max_output_tokens": 1024}
response, _ = gemini_chat.chat(system_message, history, gen_conf)
print(response)
