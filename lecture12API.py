import openai

openai.api_key=""

def Senitment_analysis(text):
    messages = [
        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given text.
                                        If you're unsure of an answer, you can say "not sure" and recommend users to review manually."""},
        {"role": "user", "content": f"""Analyze the following text and determine if the sentiment is: positive or negative.
                                        Return answer in single word as either positive or negative: {text}"""}
        ]
    
    response=openai.chat.completions.create(model='gpt-3.5-turbo',
                                        messages=messages,
                                        max_tokens=1,
                                        n=1,
                                        temperature=0)
    response_text=response.choices[0].message.content.strip().lower()

    return response_text

# calling the function
input='i hate fast food'
response=Senitment_analysis(input)
# print(input,': the sentiment is', response)

Senitment_analysis('i hate oils')

# GENERATING BLOG POST
def generate_blog(topic):
    messages=[
        {"role":"system","content":"""You are trained to analyze a topic and generate a blog post.
          The blog post must contain 1500 to 3000 words (No less than 1500 words)."""},
        {"role":"user","content":f"""Analyze the topic and generate a blog post. the topic is {topic} 
         The blog post should contain the following format.
                                        1) Title (Not more than one line).
                                        2) Introduction (Give introducion about the topic)
                                        3) Add an image url relevent to the topic.
                                        4) Add 2/3 subheadings and explain them.
                                        5) Body (should describe the facts and findings)
                                        6) Add an image url relevent to the topic.
                                        7) Add 2/3 subheadings and explain them.
                                        8) General FAQ regarding the topic.
                                        9) Conclusion of the topic."""}
    ]


    response=openai.chat.completions.create(model='gpt-3.5-turbo-16k',
                                        messages=messages,
                                        max_tokens=3000,
                                        n=1,
                                        temperature=0.5)
    response_text=response.choices[0].message.content.strip().lower()
    return response_text

user_input ="Machine Learning"
blog = generate_blog(user_input)
print(blog)