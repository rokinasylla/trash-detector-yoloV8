import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import gdown

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Poubelles YOLOv8",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .download-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#  CONFIGURATION DU MODÈLE

MODEL_GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1jfH0da0ALkH7qPW0ZyYY5MIC_NfIIBrq"
MODEL_PATH = "best.pt"

#  FONCTION DE TÉLÉCHARGEMENT DU MODÈLE 
@st.cache_resource
def download_and_load_model():
    """Télécharge le modèle depuis Google Drive et le charge"""
    try:
        # Vérifier si le modèle existe déjà
        if not os.path.exists(MODEL_PATH):
            st.info("📥 Téléchargement du modèle... (première utilisation)")
            
            # Télécharger depuis Google Drive
            gdown.download(MODEL_GDRIVE_URL, MODEL_PATH, quiet=False)
            
            if not os.path.exists(MODEL_PATH):
                st.error("❌ Échec du téléchargement du modèle depuis Google Drive")
                return None
        
        # Charger le modèle YOLO
        model = YOLO(MODEL_PATH)
        return model
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None

# CHARGEMENT AUTOMATIQUE DU MODÈLE
if 'model' not in st.session_state or st.session_state.model is None:
    with st.spinner("🔄 Chargement du modèle YOLOv8..."):
        st.session_state.model = download_and_load_model()
        if st.session_state.model is not None:
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False

# Titre principal
st.markdown('<p class="main-header">🗑️ Détecteur de Poubelles YOLOv8</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Détection automatique - Pleines ou Vides</p>', unsafe_allow_html=True)

# Message de statut du modèle
if st.session_state.model_loaded:
    st.success("✅ Modèle chargé et prêt à l'emploi!")
else:
    st.error("❌ Impossible de charger le modèle. Veuillez vérifier la configuration.")
    st.stop()

# Sidebar - Configuration et Téléchargement
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Informations du modèle
    st.markdown("### 📊 Informations du Modèle")
    if hasattr(st.session_state.model, 'names'):
        st.write(f"**Classes:** {len(st.session_state.model.names)}")
        classes_list = list(st.session_state.model.names.values())
        for idx, class_name in enumerate(classes_list):
            st.write(f"  {idx}: {class_name}")
    
    st.markdown("---")
    
    # SECTION TÉLÉCHARGEMENT DU MODÈLE
    st.markdown("### 📥 Télécharger le Modèle")
    st.info("Vous pouvez télécharger notre modèle pour l'utiliser hors ligne ou l'intégrer dans votre propre application.")
    
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            btn = st.download_button(
                label="⬇️ Télécharger best.pt",
                data=file,
                file_name="best.pt",
                mime="application/octet-stream",
                use_container_width=True
            )
            if btn:
                st.success("✅ Téléchargement lancé!")
    else:
        st.warning("⚠️ Modèle non disponible pour le téléchargement")
    
    # Informations sur l'utilisation du modèle
    with st.expander("ℹ️ Comment utiliser le modèle téléchargé"):
        st.markdown("""
        ### Utilisation en Python:
        
        ```python
        from ultralytics import YOLO
        
        # Charger le modèle
        model = YOLO('best.pt')
        
        # Prédiction sur une image
        results = model.predict('image.jpg')
        
        # Afficher les résultats
        results[0].show()
        ```
        
        ### Détection vidéo:
        
        ```python
        model = YOLO('best.pt')
        results = model.predict('video.mp4', save=True)
        ```
        
        ### Webcam en temps réel:
        
        ```python
        import cv2
        model = YOLO('best.pt')
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            results = model.predict(frame)
            cv2.imshow('Detection', results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ```
        """)
    
    st.markdown("---")
    
    # Paramètres de détection
    st.markdown("### 🎛️ Paramètres de Détection")
    confidence = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Confidence minimale pour les détections"
    )
    
    iou_threshold = st.slider(
        "Seuil IoU (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Seuil pour la suppression des non-maxima"
    )
    
    # Options d'affichage
    st.markdown("### 🎨 Affichage")
    show_labels = st.checkbox("Afficher les labels", value=True)
    show_conf = st.checkbox("Afficher la confiance", value=True)
    box_thickness = st.slider("Épaisseur des boîtes", 1, 5, 2)

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📸 Images", "🎥 Vidéos", "ℹ️ À Propos"])

# TAB 1: DETECTION SUR IMAGE
with tab1:
    st.header("📸 Détection sur Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image d'entrée")
        uploaded_image = st.file_uploader(
            "Choisissez une image de poubelle",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Image originale", use_container_width=True)
    
    with col2:
        st.subheader("Résultat de détection")
        
        if uploaded_image is not None:
            try:
                with st.spinner("🔍 Analyse en cours..."):
                    # Convertir en array numpy
                    img_array = np.array(image)
                    
                    # Détection
                    results = st.session_state.model.predict(
                        img_array,
                        conf=confidence,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    # Image annotée
                    annotated = results[0].plot(
                        line_width=box_thickness,
                        labels=show_labels,
                        conf=show_conf
                    )
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_rgb, caption="Détections", use_container_width=True)
                    
                    # Statistiques
                    boxes = results[0].boxes
                    num_detections = len(boxes)
                    
                    st.markdown(f"""
                    <div class="stat-box">
                        <h2>{num_detections}</h2>
                        <p>Objets détectés</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Détails des détections
                    if num_detections > 0:
                        with st.expander("📊 Détails des détections"):
                            for i, box in enumerate(boxes):
                                cls = int(box.cls[0])
                                conf_val = float(box.conf[0])
                                label = st.session_state.model.names[cls]
                                
                                # Coordonnées de la boîte
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                st.write(f"**Objet {i+1}:**")
                                st.write(f"  - Classe: {label}")
                                st.write(f"  - Confiance: {conf_val:.2%}")
                                st.write(f"  - Position: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
                                st.markdown("---")
                        
                        # Télécharger l'image annotée
                        from io import BytesIO
                        
                        result_img = Image.fromarray(annotated_rgb)
                        buf = BytesIO()
                        result_img.save(buf, format='JPEG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Télécharger l'image avec détections",
                            data=buf,
                            file_name="detection_result.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection: {str(e)}")
        else:
            st.info("👆 Uploadez une image pour commencer la détection")

#  TAB 2: DETECTION SUR VIDEOS 
with tab2:
    st.header("🎥 Détection sur Vidéos")
    
    uploaded_video = st.file_uploader(
        "Choisissez une vidéo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Sauvegarder la vidéo temporairement
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vidéo originale")
            st.video(uploaded_video)
        
        with col2:
            st.subheader("Traitement")
            
            if st.button("▶️ Lancer la détection", use_container_width=True, type="primary"):
                try:
                    # Ouvrir la vidéo
                    cap = cv2.VideoCapture(video_path)
                    
                    # Propriétés de la vidéo
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"📹 Vidéo: {width}x{height} @ {fps} FPS - {total_frames} frames")
                    
                    # Fichier de sortie
                    output_path = tempfile.mktemp(suffix='.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # Barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    frame_placeholder = st.empty()
                    
                    frame_count = 0
                    detection_stats = []
                    start_time = time.time()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Détection sur le frame
                        results = st.session_state.model.predict(
                            frame,
                            conf=confidence,
                            iou=iou_threshold,
                            verbose=False
                        )
                        
                        # Annoter le frame
                        annotated_frame = results[0].plot(
                            line_width=box_thickness,
                            labels=show_labels,
                            conf=show_conf
                        )
                        
                        # Sauvegarder
                        out.write(annotated_frame)
                        
                        # Statistiques
                        num_detections = len(results[0].boxes)
                        detection_stats.append(num_detections)
                        
                        # Mise à jour de l'affichage (tous les 10 frames)
                        frame_count += 1
                        if frame_count % 10 == 0:
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            
                            elapsed = time.time() - start_time
                            fps_current = frame_count / elapsed if elapsed > 0 else 0
                            eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                            
                            status_text.text(
                                f"⏳ Frame {frame_count}/{total_frames} ({progress:.1%}) | "
                                f"Vitesse: {fps_current:.1f} FPS | "
                                f"ETA: {eta:.0f}s"
                            )
                            
                            # Afficher le frame actuel
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Libérer les ressources
                    cap.release()
                    out.release()
                    
                    progress_bar.progress(1.0)
                    processing_time = time.time() - start_time
                    status_text.text(f"✅ Traitement terminé en {processing_time:.1f}s!")
                    
                    # Afficher la vidéo traitée
                    st.success("🎉 Détection terminée!")
                    
                    # Lire le fichier vidéo traité
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    st.download_button(
                        label="📥 Télécharger la vidéo traitée",
                        data=video_bytes,
                        file_name="video_detectee.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.video(video_bytes)
                    
                    # Statistiques globales
                    if detection_stats:
                        avg_detections = np.mean(detection_stats)
                        max_detections = np.max(detection_stats)
                        total_detections = sum(detection_stats)
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>{total_detections}</h3>
                                <p>Détections totales</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_stat2:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>{avg_detections:.1f}</h3>
                                <p>Moyenne/frame</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_stat3:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>{max_detections}</h3>
                                <p>Maximum/frame</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Nettoyer les fichiers temporaires
                    try:
                        cap.release()
                        time.sleep(0.5)  # Attendre un peu
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except Exception as cleanup_error:
                        # Ignorer les erreurs de nettoyage
                        pass
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    if 'cap' in locals():
                        cap.release()
                    # Nettoyer en cas d'erreur
                    try:
                        if 'video_path' in locals() and os.path.exists(video_path):
                            os.unlink(video_path)
                    except:
                        pass
    else:
        st.info("👆 Uploadez une vidéo pour commencer")

# AB 3: À PROPOS 
with tab3:
    st.header("ℹ️ À Propos de l'Application")
    
    st.markdown("""
    ### 🎯 Objectif
    Cette application utilise l'intelligence artificielle (YOLOv8) pour détecter automatiquement 
    si les poubelles sont **pleines** ou **vides** sur des images et vidéos.
    
    ### 🚀 Fonctionnalités
    - ✅ Détection instantanée sur images
    - ✅ Traitement de vidéos complètes
    - ✅ Téléchargement des résultats annotés
    - ✅ Téléchargement du modèle pour utilisation hors ligne
    - ✅ Paramètres ajustables en temps réel
    
    ### 🛠️ Technologies Utilisées
    - **YOLOv8** (Ultralytics) - Détection d'objets
    - **Streamlit** - Interface web
    - **OpenCV** - Traitement vidéo
    - **Python** - Langage de programmation
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📥 Télécharger le Modèle
    
    Vous pouvez télécharger notre modèle entraîné depuis la **barre latérale** (Sidebar) 
    pour l'utiliser dans vos propres projets Python.
    
    #### Cas d'usage :
    - 🔬 Recherche et développement
    - 📱 Intégration dans une application mobile
    - 🖥️ Utilisation hors ligne
    - 🎓 Projets éducatifs
    - 🏭 Déploiement en production
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Performance du Modèle
        
        Le modèle a été entraîné sur un dataset de poubelles avec:
        - Images d'entraînement diversifiées
        - Différents angles et éclairages
        - Multiples types de poubelles
        """)
    
    with col2:
        st.markdown("""
        ### 🎓 Comment Utiliser
        
        1. Choisissez l'onglet **Images** ou **Vidéos**
        2. Uploadez votre fichier
        3. Ajustez les paramètres si nécessaire
        4. Visualisez les résultats
        5. Téléchargez les fichiers annotés
        """)
    
    st.markdown("---")
    
    # Documentation technique
    with st.expander("📚 Documentation Technique"):
        st.markdown("""
        ### Architecture du Modèle
        
        **YOLOv8** (You Only Look Once version 8) est un modèle de détection d'objets 
        en temps réel de pointe qui offre:
        
        - **Vitesse**: Détection ultra-rapide (> 30 FPS)
        - **Précision**: Haute performance de détection
        - **Efficacité**: Optimisé pour CPU et GPU
        
        ### Classes Détectées
        
        Le modèle peut identifier les classes suivantes:
        """)
        
        if hasattr(st.session_state.model, 'names'):
            for idx, name in st.session_state.model.names.items():
                st.write(f"- **Classe {idx}**: {name}")
        
        st.markdown("""
        ### Paramètres de Détection
        
        - **Confidence**: Seuil de confiance minimum (0-1)
        - **IoU (Intersection over Union)**: Seuil pour supprimer les détections dupliquées
        - **Line Thickness**: Épaisseur des boîtes de détection
        """)
    
    st.markdown("---")
    
    st.info("💡 **Astuce**: Pour de meilleurs résultats, utilisez des images bien éclairées et des vidéos stables.")
    
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #666;'>
        <p><strong>🗑️ Détecteur de Poubelles YOLOv8</strong></p>
        <p>Développé avec ❤️ using Streamlit & Ultralytics YOLO</p>
        <p><small>Version 1.0 - 2024</small></p>
    </div>

    """, unsafe_allow_html=True)

