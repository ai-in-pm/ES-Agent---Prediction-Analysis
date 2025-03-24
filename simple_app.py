import streamlit as st
import traceback

# Configure the page
st.set_page_config(page_title="ES-Agent: Diagnostic Mode", page_icon="📊", layout="wide")

# Title and explanation
st.title("ES-Agent: Diagnostic Mode")
st.markdown("This is a simplified version of the app to diagnose issues.")

# Try to import the utility modules with detailed error reporting
try:
    st.subheader("Importing ES Calculations Module")
    try:
        from utils.es_calculations import EarnedScheduleCalculator
        st.success("✅ Successfully imported EarnedScheduleCalculator")
    except Exception as e:
        st.error(f"❌ Error importing EarnedScheduleCalculator: {str(e)}")
        st.code(traceback.format_exc())
    
    st.subheader("Importing Prediction Models Module")
    try:
        from utils.prediction_models import ESPredictionModels
        st.success("✅ Successfully imported ESPredictionModels")
        
        # Check if pmdarima is available
        try:
            from utils.prediction_models import PMDARIMA_AVAILABLE
            if PMDARIMA_AVAILABLE:
                st.success("✅ pmdarima is available for advanced forecasting")
            else:
                st.warning("⚠️ pmdarima is not available - some forecasting features will be limited")
        except Exception as e:
            st.warning("⚠️ Could not determine pmdarima availability")
            
    except Exception as e:
        st.error(f"❌ Error importing ESPredictionModels: {str(e)}")
        st.code(traceback.format_exc())
    
    st.subheader("Importing Data Processing Module")
    try:
        from utils.data_processing import ESDataProcessor
        st.success("✅ Successfully imported ESDataProcessor")
    except Exception as e:
        st.error(f"❌ Error importing ESDataProcessor: {str(e)}")
        st.code(traceback.format_exc())
        
except Exception as e:
    st.error(f"General import error: {str(e)}")
    st.code(traceback.format_exc())

# Simple UI elements to test Streamlit functionality
st.subheader("Testing Basic Streamlit UI Components")

st.write("This section tests if Streamlit UI components are working properly")

try:
    # Test a few basic UI components
    test_input = st.text_input("Test text input")
    test_slider = st.slider("Test slider", 0, 100, 50)
    test_button = st.button("Test button")
    
    # Display the values
    st.write(f"Text input value: {test_input}")
    st.write(f"Slider value: {test_slider}")
    
    if test_button:
        st.write("Button was clicked!")
        
    st.success("✅ Basic UI components are working")
    
except Exception as e:
    st.error(f"❌ Error in UI components: {str(e)}")
    st.code(traceback.format_exc())

# Simple sidebar
with st.sidebar:
    st.header("Diagnostic Sidebar")
    st.write("This sidebar tests if the Streamlit layout is working correctly.")
    
    # Test file uploader
    st.subheader("Test File Upload")
    uploaded_file = st.file_uploader("Upload a test file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
