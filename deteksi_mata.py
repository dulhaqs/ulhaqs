import streamlit as st  # Library untuk membuat antarmuka web
import cv2  # Library untuk pengolahan gambar
import numpy as np  # Library untuk komputasi numerik
import time  # Library untuk mengelola waktu
from keras.models import load_model  # Library untuk memuat model Keras
from keras.preprocessing import image  # Library untuk pemrosesan gambar
from PIL import Image, ImageOps  # Library untuk manipulasi gambar
from streamlit_option_menu import option_menu  # Library untuk membuat option menu

# Menetapkan ikon dan judul halaman
icon_path = "👀"
st.set_page_config(page_title="Deteksi Katarak dan Glaukoma.AI", page_icon=icon_path)

# Memuat model yang telah dilatih
model = load_model('compressed_model_deteksikatarakglaukoma.h5')

def load_and_process_image(img_path, target_size=(644, 426)):
    """Memuat dan memproses gambar"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array /= 255.0  # Normalisasi
    return img_array

def colored_divider(color, thickness="15px"):
    st.markdown(f"<hr style='border: none; border-top: {thickness} solid {color};' />", unsafe_allow_html=True)

def large_title(text, size="40px"):
    st.markdown(f"<h1 style='font-size: {size};'>{text}</h1>", unsafe_allow_html=True)

def large_text(text, size="20px"):
    st.markdown(f"<p style='font-size: {size};'>{text}</p>", unsafe_allow_html=True)

def predict_image_class(model, img_path, class_labels):
    """Melakukan prediksi kelas gambar menggunakan model"""
    processed_image = load_and_process_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    skor_kepercayaan = float(prediction[0][predicted_class])
    return predicted_label, skor_kepercayaan

# Daftar label kelas (pastikan urutannya sesuai dengan class_indices pada training)
class_labels = ['Glaukoma', 'Katarak','Normal']

# Aplikasi Streamlit
# Navigasi menggunakan option menu
with st.sidebar:
    halaman_terpilih = option_menu(
        "Pilih Halaman",
        ["Beranda", "Tentang", "Halaman Deteksi", "Contact Us"],
        icons=["house", "info-circle", "camera", "phone"],
        menu_icon="cast",
        default_index=0
    )

if halaman_terpilih == "Beranda":
    # Tampilkan halaman beranda
    large_title("Selamat Datang di Aplikasi Deteksi Katarak dan Glaukoma")
    colored_divider(color="blue")  # Menambahkan garis pemisah berwarna
    st.write(
        "Aplikasi ini memungkinkan Anda untuk mendeteksi gangguan mata, "
        "pada aplikasi ini Anda dapat mendeteksi gangguan yang dialami pada mata berdasarkan gambar syaraf mata."
    )
    st.write(
        "Silahkan pilih halaman deteksi untuk melanjutkan pemeriksaan atau pilih halaman visualisasi model untuk melihat hasil kinerja model AI, seperti accuracy class, confusional matrix, accuracy epoch dan loss epoch."
    )
elif halaman_terpilih == "Tentang":
    large_title("APA ITU PENYAKIT MATA KATARAK ?")
    colored_divider(color="red")
    st.image("gambar_katarak.jpg")
    large_text("Katarak adalah gangguan penglihatan pada mata yang disebabkan adanya keruhan pada lensa mata. Pada bola mata, kekeruhan ini akan menutupi masuknya cahaya ke mata, sehingga mengakibatkan penurunan pada penglihatan. Pada awalnya orang yang memiliki katarak terjadi gumpalan kecil pada mata yang tidak mengganggu penglihatan, namun semakin lama dibiarkan maka gumpalan tersebut perlahan semakin besar dan akan terjadinya penuruan pada ketajaman mata.")
    large_title("APA ITU PENYAKIT MATA GLAUKOMA ?")
    colored_divider(color="green")
    st.image("glaukoma_gambar.png")
    large_text("glaukoma adalah penyakit mata yang dapat menyebabkan kebutaan, namun berbeda dengan katarak yang masih bisa disembuhkan melalui operasi. glaukoma merupakan penyakit mata yang berjalan secara progresif, hal ini menyebabkan gejala penyakit glaukoma tidak dirasakan oleh penderitanya dan penyakit ini bersifat permanen atau tidak dapat diperbaiki (irreversible) meskipun dengan jalan operasi")

elif halaman_terpilih == "Halaman Deteksi":
    # Tampilkan halaman deteksi
    st.title("Unggah Gambar")
    colored_divider(color="green")
    st.markdown("---")

    # Unggah gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silakan pilih gambar", type=["jpg", "jpeg", "png"])
    
    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang diunggah", use_column_width=True)
        if st.button("Mulai Deteksi"):
            with st.spinner('Medeteksi gambar...'):
                # Simpan berkas gambar yang diunggah ke lokasi sementara
                with open("temp_image.jpg", "wb") as f:
                    f.write(berkas_gambar.getbuffer())

                # Lakukan prediksi pada berkas yang disimpan
                predicted_label, skor_kepercayaan = predict_image_class(model, "temp_image.jpg", class_labels)

            # Tampilkan hasil prediksi
            st.write(f"Hasil Deteksi : {predicted_label}")
            st.write(f"Skor Kepercayaan: {skor_kepercayaan * 100:.2f}%")

            # Tampilkan pesan berdasarkan hasil prediksi
            if predicted_label == 'Normal':
                st.write("Berdasarkan hasil deteksi yang telah dilakukan berdasarkan gambar syaraf mata, anda dinyatakan NORMAL. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami.")
            elif predicted_label == 'Katarak':
                st.write("Berdasarkan hasil deteksi yang telah dilakukan berdasarkan gambar syaraf mata, anda terdeteksi adanya KATARAK pada mata. Lakukan tindakan segera dengan dokter untuk mencegah perburukan pada syaraf mata.")
            elif predicted_label == 'Glaukoma':
                st.write("Berdasarkan Hasil deteksi yang telah dilakukan berdasarkan gambar syaraf mata, anda terdeteksi ancaman penyakit mata GLAUKOMA, lakukan pemeriksaan lebih lanjut dengan dokter spesialis mata mencegah terjadinya kebutaan pada mata.")

elif halaman_terpilih == "Contact Us":
    large_title("Contact Us:")
    contact_option = option_menu(
        menu_title="Select Contact",
        options=["WhatsApp", "Email", "Instagram"],
        icons=["whatsapp", "envelope", "instagram"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#FAFAFA", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#575757",
            },
            "nav-link-selected": {"background-color": "#584BFF"},
        }
    )

    if contact_option == "WhatsApp":
        st.markdown(
            """
            <div style='text-align: center; color: white;'>
                <a href='https://wa.me/089619556098' target='_blank' style='text-decoration: none; color: white;'>
                    <img src='https://img.icons8.com/ios-filled/50/FAFAFA/whatsapp.png' alt='WhatsApp' style='width: 24px; height: 24px; margin-right: 8px; vertical-align: middle;'>
                    WhatsApp: 089619556098
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif contact_option == "Email":
        st.markdown(
            """
            <div style='text-align: center; color: white;'>
                <a href='https://mail.google.com/mail/?view=cm&fs=1&to=dhiyaulhaq98841@gmail.com' target='_blank' style='text-decoration: none; color: white;'>
                    <img src='https://img.icons8.com/ios-filled/50/FAFAFA/new-post.png' alt='Email' style='width: 24px; height: 24px; margin-right: 8px; vertical-align: middle;'>
                    Email: dhiyaulhaq98841@gmail.com
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif contact_option == "Instagram":
        st.markdown(
            """
            <div style='text-align: center; color: white;'>
                <a href='https://www.instagram.com/ulhaaq_s' target='_blank' style='text-decoration: none; color: white;'>
                    <img src='https://img.icons8.com/ios-filled/50/FAFAFA/instagram-new.png' alt='Instagram' style='width: 24px; height: 24px; margin-right: 8px; vertical-align: middle;'>
                    Instagram: @ulhaaq_s
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
