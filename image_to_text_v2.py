import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import requests
from PIL import Image
import os
from dotenv import load_dotenv
import pytesseract

# Load environment variables (for API key)
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key (first 5 characters): {api_key[:5] if api_key else 'Not set'}")

if not api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Function to extract text from image using OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Function to process the text with OpenAI API
def process_text_with_openai(text):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {"error": {"message": "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."}}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial assistant. Extract the following information from the given text: Security name, Ticker symbol (if available), Number of shares or units, Currency, and Current market value. Format the data as a table, excluding any historical data, transaction history, or irrelevant information. Flag any unclear or potentially misread entries."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": 500
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": {"message": f"API request failed: {str(e)}"}}

# Function to parse the OpenAI response and create a DataFrame
def parse_openai_response(response):
    try:
        content = response['choices'][0]['message']['content']
        # Parse the content into a DataFrame
        lines = content.split('\n')
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        data = []
        for line in lines[2:]:
            if '|' in line:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(row) == len(headers):
                    data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        return df, None  # Return DataFrame and no error message
    except KeyError as e:
        error_message = f"Error: Invalid API response structure. Missing key: {str(e)}"
        return None, error_message  # Return no DataFrame and error message
    except Exception as e:
        error_message = f"Error processing API response: {str(e)}"
        return None, error_message  # Return no DataFrame and error message

# Function to download DataFrame as Excel
def download_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Login form
def login_form():
    st.subheader("Login")
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    return user_id, password

# Login validation
def validate_login(user_id, password):
    return user_id == 'kristaldemo' and password == 'r)w6nREP'

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Streamlit app
def main():
    st.title("Bank Statement Image Processor")

    if not st.session_state.logged_in:
        user_id, password = login_form()

        if st.button("Login"):
            if validate_login(user_id, password):
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid user ID or password. Please try again.")
    else:
        st.write("Upload a bank statement image to extract financial information.")

        uploaded_file = st.file_uploader("Choose a bank statement image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    # Extract text from image
                    extracted_text = extract_text_from_image(uploaded_file)
                    
                    # Process the extracted text
                    response = process_text_with_openai(extracted_text)
                    
                    if "error" in response:
                        st.error(f"API Error: {response['error']['message']}")
                        st.json(response)  # Display the raw response for debugging
                    else:
                        # Parse the response and create DataFrame
                        df, error_message = parse_openai_response(response)
                        
                        if error_message:
                            st.error(error_message)
                            st.json(response)  # Display the raw response for debugging
                        elif df is not None:
                            # Display the results
                            st.subheader("Extracted Financial Information")
                            st.dataframe(df)
                            
                            # Summary
                            st.subheader("Summary")
                            st.write(f"Total number of holdings extracted: {len(df)}")
                            flagged_entries = df[df.apply(lambda row: 'unclear' in ' '.join(row).lower(), axis=1)]
                            if not flagged_entries.empty:
                                st.warning(f"Number of potentially unclear entries: {len(flagged_entries)}")
                                st.write("Please review the following entries manually:")
                                st.dataframe(flagged_entries)
                            
                            # Download button
                            excel_data = download_excel(df)
                            st.download_button(
                                label="Download Excel file",
                                data=excel_data,
                                file_name="extracted_financial_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.error("Unable to process the image. Please try again.")

        st.write("Note: This app uses OpenAI's API for text processing. Ensure you have a valid API key set up in your environment variables.")

        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.success("Logged out successfully!")

if __name__ == "__main__":
    main()