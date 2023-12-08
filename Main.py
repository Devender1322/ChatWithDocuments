import streamlit as st
from pages import PDF, CSV
from streamlit import runtime
from PIL import Image
# from keras.preprocessing import image

st.set_page_config(page_title="File analyzer ", page_icon= "📂", layout="wide" )

def resize_image(image_path, target_size=(2180, 2180)):
    image = Image.open(image_path)
    resized_image = image.resize(target_size)
    return resized_image

# Function to create a session state
def create_session_state():
    class SessionState:
        current_page = "home"

    return SessionState()

# Create a session state
session_state = create_session_state()

# Home page (main.py)
if session_state.current_page == "home":
    st.title("AI Empowered file Analyzer App")
    st.subheader("Explore the Magic of AI File Analyzer")
    st.write("This is the homepage briefing the capabilities of AI driven File Analyzer")
    col1, col2, col3 = st.columns(3)
    image_path = r"D:\File analyzer\image\fotor-ai-20231201173549.jpg"

    # Resize the image
    # resized_image = resize_image(image_path)

    resized_image = resize_image(image_path)
    col1.image(resized_image, use_column_width=True, caption="PDF CSV DOCS TXT")
  

    # col1.image (image_path, use_column_width=True)

    # Column 2 - Text
    # col2.write("Streamlit is an amazing tool for creating web apps with Python.")
    col2.write("Explore its features and communicate with your Documents!")

    # Column 3 - Contact Information
    col3.write("Contact Information:")
    col3.write("Email: contact@example.com")
    col3.write("Linkedin:")

    # Adding a Divider
    st.markdown("---")
   

    # 

    # Button to navigate to PDF page
    # if st.button("Go to PDF Page"):
    #     session_state.current_page = "pdf"

    # # Button to navigate to CSV page
    # if st.button("Go to CSV Page"):
    #     session_state.current_page = "csv"

# PDF page (pages/pdf.py)
elif session_state.current_page == "pdf":
    # pdf.show()
    st.button("Go back to Home", on_click=lambda: session_state.__setattr__("current_page", "home"))

# CSV page (pages/csv.py)
elif session_state.current_page == "csv":
    # csv.show()
    st.button("Go back to Home", on_click=lambda: session_state.__setattr__("current_page", "home"))



    

    # Add an interesting feature
    
    

if __name__ == "__main__":
    create_session_state()

