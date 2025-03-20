import os
import streamlit as st
from PIL import Image
import numpy as np
import time
from pathlib import Path
import pandas as pd
import threading
import queue
import concurrent.futures
import io
import tkinter as tk
from tkinter import filedialog

class ImageConverter:
    def __init__(self):
        self.setup_variables()
        
    def setup_variables(self):
        self.is_processing = False
        self.dpcm_results = {}
        self.total_jpeg_dpcm = 0
        self.total_webp_dpcm = 0
        self.successful_dpcm_count = 0
        self.total_original_size = 0
        self.total_jpeg_size = 0
        self.total_webp_size = 0
        self.finished_count = 0
        self.progress = 0
        
    def calculate_dpcm(self, original_path, jpeg_path, webp_path):
        try:
            original_img = np.array(Image.open(original_path).convert('L'))
            jpeg_img = np.array(Image.open(jpeg_path).convert('L'))
            webp_img = np.array(Image.open(webp_path).convert('L'))
            
            jpeg_correlation = np.corrcoef(original_img.flat, jpeg_img.flat)[0, 1]
            webp_correlation = np.corrcoef(original_img.flat, webp_img.flat)[0, 1]
            
            return jpeg_correlation, webp_correlation
        except Exception:
            return (0, 0)
    
    def process_image(self, img_path, jpeg_folder, webp_folder, quality, calc_dpcm, total_images):
        try:
            img_filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(img_filename)[0]
            
            original_size = os.path.getsize(img_path) / 1024
            self.total_original_size += original_size
            
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            jpeg_path = os.path.join(jpeg_folder, f"{name_without_ext}.jpg")
            img.save(jpeg_path, 'JPEG', quality=quality)
            
            webp_path = os.path.join(webp_folder, f"{name_without_ext}.webp")
            img.save(webp_path, 'WEBP', quality=quality)
            
            jpeg_size = os.path.getsize(jpeg_path) / 1024
            webp_size = os.path.getsize(webp_path) / 1024
            
            self.total_jpeg_size += jpeg_size
            self.total_webp_size += webp_size
            
            jpeg_size_diff = original_size - jpeg_size
            webp_size_diff = original_size - webp_size
            
            jpeg_size_percent = (jpeg_size_diff / original_size) * 100 if original_size > 0 else 0
            webp_size_percent = (webp_size_diff / original_size) * 100 if original_size > 0 else 0
            
            result = {
                "original_size_kb": round(original_size, 2),
                "jpeg_size_kb": round(jpeg_size, 2),
                "webp_size_kb": round(webp_size, 2),
                "jpeg_size_reduction_kb": round(jpeg_size_diff, 2),
                "webp_size_reduction_kb": round(webp_size_diff, 2),
                "jpeg_size_reduction_percent": round(jpeg_size_percent, 2),
                "webp_size_reduction_percent": round(webp_size_percent, 2),
            }
            
            if calc_dpcm:
                jpeg_corr, webp_corr = self.calculate_dpcm(img_path, jpeg_path, webp_path)
                jpeg_corr = round(jpeg_corr, 5)
                webp_corr = round(webp_corr, 5)
                
                result.update({
                    "jpeg_correlation": jpeg_corr,
                    "webp_correlation": webp_corr,
                })
                
                if jpeg_corr > 0 and webp_corr > 0:
                    jpeg_diff_percent = (1 - jpeg_corr) * 100
                    webp_diff_percent = (1 - webp_corr) * 100
                    
                    result.update({
                        "jpeg_diff_percent": round(jpeg_diff_percent, 5),
                        "webp_diff_percent": round(webp_diff_percent, 5)
                    })
                    
                    self.total_jpeg_dpcm += jpeg_corr
                    self.total_webp_dpcm += webp_corr
                    self.successful_dpcm_count += 1
            
            self.dpcm_results[img_filename] = result
            self.finished_count += 1
            self.progress = (self.finished_count / total_images) * 100
            
            return f"Converted: {img_filename}"
            
        except Exception as e:
            return f"Error converting {img_path}: {str(e)}"
    
    def convert_images(self, folder_path, quality, calc_dpcm):
        try:
            self.setup_variables()  # Reset variables for new conversion
            
            jpeg_folder = os.path.join(folder_path, "jpeg_output")
            webp_folder = os.path.join(folder_path, "webp_output")
            
            os.makedirs(jpeg_folder, exist_ok=True)
            os.makedirs(webp_folder, exist_ok=True)
            
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
                image_files.extend(list(Path(folder_path).glob(f"*{ext.upper()}")))
            
            total_images = len(image_files)
            
            if total_images == 0:
                return "No image files found in the selected folder.", None, 0
            
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            result_area = st.empty()
            
            status_placeholder.text(f"Found {total_images} images. Starting conversion...")
            
            # Results container to show progress in real-time
            result_container = st.container()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Submit all image processing tasks
                future_to_img = {
                    executor.submit(
                        self.process_image, img_path, jpeg_folder, webp_folder, quality, calc_dpcm, total_images
                    ): img_path for img_path in image_files
                }
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_img)):
                    result = future.result()
                    with result_container:
                        st.write(result)
                    
                    # Update progress bar
                    progress_bar.progress(self.progress / 100)
                    status_placeholder.text(f"Processing... {self.finished_count}/{total_images} images completed")
            
            # Create Excel results if we have data
            excel_path = None
            summary_data = None
            
            if self.dpcm_results:
                excel_path = os.path.join(folder_path, "conversion_results.xlsx")
                
                avg_jpeg_dpcm = round(self.total_jpeg_dpcm / self.successful_dpcm_count, 5) if self.successful_dpcm_count > 0 else 0
                avg_webp_dpcm = round(self.total_webp_dpcm / self.successful_dpcm_count, 5) if self.successful_dpcm_count > 0 else 0
                
                avg_jpeg_diff_percent = round((1 - avg_jpeg_dpcm) * 100, 5) if avg_jpeg_dpcm > 0 else 0
                avg_webp_diff_percent = round((1 - avg_webp_dpcm) * 100, 5) if avg_webp_dpcm > 0 else 0
                
                total_jpeg_diff = self.total_original_size - self.total_jpeg_size
                total_webp_diff = self.total_original_size - self.total_webp_size
                
                total_jpeg_diff_percent = (total_jpeg_diff / self.total_original_size) * 100 if self.total_original_size > 0 else 0
                total_webp_diff_percent = (total_webp_diff / self.total_original_size) * 100 if self.total_original_size > 0 else 0
                
                summary_data = {
                    "Metric": [
                        "Total Images",
                        "Total Original Size (KB)",
                        "Total JPEG Size (KB)",
                        "Total WebP Size (KB)",
                        "Total JPEG Size Reduction (KB)",
                        "Total WebP Size Reduction (KB)",
                        "Total JPEG Size Reduction (%)",
                        "Total WebP Size Reduction (%)",
                        "Images with DPCM Calculation",
                        "Average JPEG Correlation",
                        "Average WebP Correlation",
                        "Average JPEG Difference (%)",
                        "Average WebP Difference (%)"
                    ],
                    "Value": [
                        total_images,
                        round(self.total_original_size, 2),
                        round(self.total_jpeg_size, 2),
                        round(self.total_webp_size, 2),
                        round(total_jpeg_diff, 2),
                        round(total_webp_diff, 2),
                        round(total_jpeg_diff_percent, 2),
                        round(total_webp_diff_percent, 2),
                        self.successful_dpcm_count,
                        avg_jpeg_dpcm,
                        avg_webp_dpcm,
                        avg_jpeg_diff_percent,
                        avg_webp_diff_percent
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                individual_data = []
                for img_name, data in self.dpcm_results.items():
                    row = {"Image": img_name}
                    row.update(data)
                    individual_data.append(row)
                
                individual_df = pd.DataFrame(individual_data)
                
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    individual_df.to_excel(writer, sheet_name='Individual Results', index=False)
                    
                    workbook = writer.book
                    summary_sheet = writer.sheets['Summary']
                    
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#7077A1',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    for col_num, value in enumerate(summary_df.columns.values):
                        summary_sheet.write(0, col_num, value, header_format)
                
            status_placeholder.text(f"Conversion complete! Converted {self.finished_count} images.")
            progress_bar.progress(1.0)
            
            return f"Conversion complete! Converted {self.finished_count} images.", excel_path, self.dpcm_results
            
        except Exception as e:
            return f"Error during conversion: {str(e)}", None, 0

def browse_folder():
    """Open a folder dialog and return the selected path"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def main():
    st.set_page_config(
        page_title="Comfy Image Converter",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #2D3250;
        color: #F6F6F6;
    }
    .stButton>button {
        background-color: #7077A1;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div {
        background-color: #7077A1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🖼️ Comfy Image Converter")
    st.subheader("Convert your images to JPEG and WebP formats")
    
    # Initialize session state for folder path if it doesn't exist
    if 'folder_path' not in st.session_state:
        st.session_state.folder_path = ""
    
    # Callback for browse button
    def on_browse_click():
        # Run the folder dialog in a separate thread to avoid blocking the UI
        def browse_thread():
            path = browse_folder()
            if path:  # Check if a path was selected (not cancelled)
                st.session_state.folder_path = path
                st.experimental_rerun()  # Rerun the app to update the UI
        
        threading.Thread(target=browse_thread).start()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Text input for folder path with the session state value
        folder_path = st.text_input("Input Folder Path", 
                                   value=st.session_state.folder_path,
                                   key="folder_path_input")
        
        # Update session state when text input changes
        st.session_state.folder_path = folder_path
        
        # Browse button with callback
        browse_button = st.button("Browse Folder", key="browse_button", on_click=on_browse_click)
        
        quality = st.slider("JPEG/WebP Quality", min_value=70, max_value=100, value=85, 
                           help="Select quality level between 70-100", key="quality_slider")
        
        calc_dpcm = st.checkbox("Calculate DPCM Correlation", value=True,
                               help="Calculate correlation between original and converted images",
                               key="dpcm_checkbox")
        
        convert_button = st.button("Convert Images", type="primary", key="convert_button")
    
    # Main content
    if convert_button:
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Please enter a valid folder path!")
        else:
            converter = ImageConverter()
            
            with st.spinner("Converting images..."):
                status, excel_path, results = converter.convert_images(folder_path, quality, calc_dpcm)
            
            st.success(status)
            
            if excel_path and os.path.exists(excel_path):
                with open(excel_path, "rb") as file:
                    st.download_button(
                        label="Download Excel Results",
                        data=file,
                        file_name="conversion_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                if results:
                    st.subheader("Summary Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Images", len(results))
                        st.metric("Total Original Size", f"{round(converter.total_original_size, 2)} KB")
                        
                    with col2:
                        total_jpeg_diff = converter.total_original_size - converter.total_jpeg_size
                        total_webp_diff = converter.total_original_size - converter.total_webp_size
                        
                        total_jpeg_diff_percent = (total_jpeg_diff / converter.total_original_size) * 100 if converter.total_original_size > 0 else 0
                        total_webp_diff_percent = (total_webp_diff / converter.total_original_size) * 100 if converter.total_original_size > 0 else 0
                        
                        st.metric("JPEG Size Reduction", f"{round(total_jpeg_diff_percent, 2)}%")
                        st.metric("WebP Size Reduction", f"{round(total_webp_diff_percent, 2)}%")
                    
                    if calc_dpcm and converter.successful_dpcm_count > 0:
                        st.subheader("DPCM Correlation Results")
                        
                        avg_jpeg_dpcm = round(converter.total_jpeg_dpcm / converter.successful_dpcm_count, 5) if converter.successful_dpcm_count > 0 else 0
                        avg_webp_dpcm = round(converter.total_webp_dpcm / converter.successful_dpcm_count, 5) if converter.successful_dpcm_count > 0 else 0
                        
                        avg_jpeg_diff_percent = round((1 - avg_jpeg_dpcm) * 100, 5) if avg_jpeg_dpcm > 0 else 0
                        avg_webp_diff_percent = round((1 - avg_webp_dpcm) * 100, 5) if avg_webp_dpcm > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("JPEG Correlation", f"{avg_jpeg_dpcm}")
                            st.metric("JPEG Difference", f"{avg_jpeg_diff_percent}%")
                        
                        with col2:
                            st.metric("WebP Correlation", f"{avg_webp_dpcm}")
                            st.metric("WebP Difference", f"{avg_webp_diff_percent}%")
                    
                    # Show detailed results in an expander
                    with st.expander("Detailed Results"):
                        # Convert results to dataframe for display
                        detailed_data = []
                        for img_name, data in results.items():
                            row = {"Image": img_name}
                            row.update(data)
                            detailed_data.append(row)
                        
                        if detailed_data:
                            df = pd.DataFrame(detailed_data)
                            st.dataframe(df)
    else:
        # Instructions when the app starts
        st.info("""
        ### Instructions:
        1. Enter the full path to your image folder in the sidebar
        2. Adjust the quality setting if needed (default is 85)
        3. Choose whether to calculate DPCM correlation
        4. Click 'Convert Images' to start the conversion
        
        The app will create two subfolders in your selected directory:
        - jpeg_output: Contains JPEG converted images
        - webp_output: Contains WebP converted images
        
        A detailed Excel report will be generated after conversion.
        """)


if __name__ == "__main__":
    main()