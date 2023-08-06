import pandas as pd
import requests
from bs4 import BeautifulSoup
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__, template_folder='templates')

# Define your chatbot's tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# Load the dataset for course data
df = pd.read_csv('course_data.csv')

# Preprocess the course names (lowercase and remove special characters)
df['course_name'] = df['course_name'].str.lower()

# Load the dataset for course subjects
course_subjects_df = pd.read_csv('course_subjects.csv')


start_suggestion = False


# Define the list of questions and their corresponding keywords
questions = [
    ("education_level", "What is your current education level?"),
    ("field_of_study", "What is your major or field of study in your undergraduate program?"),
    ("skills", "What are your skills and areas of expertise?"),
    ("interests", "What are your interests?"),
    ("location", "What is your preferred job location?")
]

answers = {
    "education_level": "",
    "field_of_study": "",
    "skills": "",
    "interests": "",
    "location": ""
}


def perform_job_search(interests, location, field_of_study, skills):
    try:
        # Construct the URL with the provided parameters
        url = f"https://www.linkedin.com/jobs/search/?keywords={interests}+{field_of_study}&location={location}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        job_listings = soup.find_all('div', class_='base-search-card__info')

        job_details = []
        for listing in job_listings:
            title_element = listing.find('h3', class_='base-search-card__title')
            job_title = title_element.text.strip() if title_element else "N/A"

            company_element = listing.find('h4', class_='base-search-card__subtitle')
            company_name = company_element.text.strip() if company_element else "N/A"

            location_element = listing.find('span', class_='job-search-card__location')
            job_location = location_element.text.strip() if location_element else "N/A"

            link_element = listing.find('a', href=True, class_='base-search-card__full-link')
            job_link = link_element['href'] if link_element else None

            job_details.append({
                "job_title": job_title,
                "company_name": company_name,
                "job_location": job_location,
                "job_link": job_link
            })

        return job_details
    except Exception as e:
        print(f"Exception occurred: {e}")
        return []



def suggest_course(user_input):
    user_input = user_input.lower()
    matched_courses = []
    count = -1

    for keywords in df['keywords']:
        count = count + 1
        if any(keyword in user_input for keyword in keywords.split()):
            matched_courses.append(df['course_name'][count])

    return matched_courses


def suggest_subjects(suggested_courses):
    subject_dict = {}
    for course in suggested_courses:
        course_subjects = course_subjects_df.loc[course_subjects_df['course_name'].str.lower() == course.lower(), 'subjects'].tolist()
        if course_subjects:
            subject_dict[course] = course_subjects[0]
    return subject_dict


# def get_Chat_response(text):
#     # Convert the user input to lowercase for consistency
#     user_input = user_input.lower()
#
#     # Check if the user wants to exit the conversation
#     if user_input == 'exit':
#         return None, "Goodbye! Have a great day."
#
#     # Store the user's answers to the questions
#     user_answers = {}
#     for question in questions:
#         user_answers[question[0]] = input(question[1] + " ")
#
#     # Generate the course suggestions based on user input
#     course_suggestions = suggest_course(' '.join(user_answers.values()))
#
#     # Prepare the chatbot's response for course suggestions
#     response = "Based on your interests and skills, we suggest the following courses:\n"
#     if course_suggestions:
#         for i, course in enumerate(course_suggestions, start=1):
#             response += f"{i}. {course}\n"
#     else:
#         response += "Sorry, no course suggestions available."
#
#     # Suggest subjects for each course
#     subject_dict = suggest_subjects(course_suggestions)
#
#     # Add subjects to the response
#     if subject_dict:
#         response += "\nSubjects for each suggested course:\n"
#         for course, subjects in subject_dict.items():
#             response += f"{course}: {subjects}\n"
#     else:
#         response += "\nSorry, no subjects available for the suggested courses."
#
#     # Perform the job search based on user preferences
#     interests = user_answers.get("interests")
#     field_of_study = user_answers.get("field_of_study")
#     skills = user_answers.get("skills")
#     location = user_answers.get("location")
#
#     job_list = perform_job_search(interests, location, field_of_study, skills)
#
#     # Generate the job suggestions string
#     stage_prompt_job = "\nHere are some job suggestions for you:"
#     job_suggestions = ""
#     for job in job_list:
#         job_suggestions += f"\nJob Title: {job['job_title']}\n"
#         job_suggestions += f"Company: {job['company_name']}\n"
#         job_suggestions += f"Location: {job['job_location']}\n"
#         job_suggestions += "------------------------\n"
#
#     # Add job suggestions to the response
#     if job_suggestions:
#         response += stage_prompt_job + job_suggestions
#     else:
#         response += "\nSorry, no job suggestions available."
#
#     return course_suggestions, response


def chatbot_logic(answers):
    # Convert the user input to lowercase for consistency
    # user_input = user_input.lower()

    # Check if the user wants to exit the conversation
    # if user_input == 'exit':
    #     return None, "Goodbye! Have a great day."

    # Store the user's answers to the questions
    # user_answers = {}
    # for question in questions:
    #     user_answers[question[0]] = input(question[1] + " ")

    # Extract information from the user's answers
    interests = answers.get("interests")
    field_of_study = answers.get("field_of_study")
    skills = answers.get("skills")
    location = answers.get("location")

    # Perform the job search based on user preferences
    job_list = perform_job_search(interests, location, field_of_study, skills)

    # Generate the job suggestions string
    stage_prompt_job = "\nHere are some job suggestions for you:"
    job_suggestions = ""
    for job in job_list:
        job_suggestions += f"\nJob Title: {job['job_title']}\n"
        job_suggestions += f"Company: {job['company_name']}\n"
        job_suggestions += f"Location: {job['job_location']}\n"
        job_suggestions += "------------------------\n"

    # Generate the course suggestions based on user input
    course_suggestions = suggest_course(' '.join(answers.values()))

    response = "\nBased on your interests and skills, we suggest the following courses:\n"
    if course_suggestions:
        for i, course in enumerate(course_suggestions, start=1):
            response += f"{i}. {course}\n"
    else:
        response += "Sorry, no course suggestions available."
    # Suggest subjects for each course
    subject_dict = suggest_subjects(course_suggestions)
    if subject_dict:
        response += "\nSubjects for each suggested course:\n"
        for course, subjects in subject_dict.items():
            response += f"{course}: {subjects}\n"
    else:
        response += "Sorry, no subjects available for the suggested courses."

    # Prepare the chatbot's response
    response += stage_prompt_job + "\n" + job_suggestions

    return course_suggestions, response


def get_bot_response(text):
    # Let's chat for 5 lines
    chat_history_ids = None
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        global start_suggestion
        global answers
        response = None
        course_suggestions = None
        user_input = request.form['msg']
        # Process user input with your chatbot_logic function

        if start_suggestion:
            question_index = 0
            for key, value in answers.items():
                if value:
                    question_index += 1
                    continue
                else:
                    answers[key] = user_input
                    question_index += 1
                    if question_index < len(questions):
                        response = questions[question_index][1]
                    else:
                        course_suggestions, response = chatbot_logic(answers)
                        start_suggestion = False
                        for ans_key in answers.keys():
                            answers[ans_key] = ""
                    break

        if user_input.lower() == "start":
            start_suggestion = True
            question1 = questions[0][1]
            response = "Please answer the following questions: \n" + question1

        # if not response:
        #     course_suggestions, response = chatbot_logic(user_input)

        if response is None:
            response = get_bot_response(user_input)

        if response is None or user_input.lower() == 'exit':
            # If the response is None, the user chose to exit the conversation
            response = "Goodbye! Have a great day."

        # Prepare the JSON response with course suggestions and chatbot response
        data = {
            "course_suggestions": course_suggestions,
            "response": response
        }
        return jsonify(data)

    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)