import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import gdown
from io import BytesIO
import threading

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
    .live-badge {
        background: #ff0000;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ========== CONFIGURATION DU MODÈLE ==========
MODEL_GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1jfH0da0ALkH7qPW0ZyYY5MIC_NfIIBrq"
MODEL_PATH = "best.pt"

# Configuration WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ========== FONCTION DE TÉLÉCHARGEMENT DU MODÈLE ==========
@st.cache_resource
def download_and_load_model():
    """Télécharge le modèle depuis Google Drive et le charge"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("📥 Téléchargement du modèle... (première utilisation)")
            gdown.download(MODEL_GDRIVE_URL, MODEL_PATH, quiet=False)
            
            if not os.path.exists(MODEL_PATH):
                st.error("❌ Échec du téléchargement du modèle")
                return None
        
        model = YOLO(MODEL_PATH)
        return model
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None

# ========== CLASSE VIDEO PROCESSOR POUR WEBRTC ==========
class YOLOVideoProcessor(VideoProcessorBase):
    """Processeur vidéo pour la détection YOLOv8 en temps réel"""
    
    def __init__(self):
        self.model = None
        self.confidence = 0.5
        self.iou_threshold = 0.45
        self.frame_count = 0
        self.detection_count = 0
        self.fps_list = []
        self.last_time = time.time()
        self.lock = threading.Lock()
    
    def set_model(self, model):
        """Définir le modèle YOLO"""
        with self.lock:
            self.model = model
    
    def set_confidence(self, confidence):
        """Définir le seuil de confiance"""
        with self.lock:
            self.confidence = confidence
    
    def set_iou(self, iou):
        """Définir le seuil IoU"""
        with self.lock:
            self.iou_threshold = iou
    
    def recv(self, frame):
        """Traiter chaque frame de la webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        with self.lock:
            if self.model is None:
                return frame
            
            try:
                # Calculer le FPS
                current_time = time.time()
                fps = 1 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
                self.last_time = current_time
                self.fps_list.append(fps)
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)
                avg_fps = np.mean(self.fps_list)
                
                # Détection YOLOv8
                results = self.model.predict(
                    img,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Annoter l'image
                annotated_img = results[0].plot()
                
                # Compter les détections
                num_detections = len(results[0].boxes)
                self.frame_count += 1
                self.detection_count += num_detections
                
                # Ajouter des informations sur l'image
                cv2.putText(annotated_img, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_img, f"Detections: {num_detections}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Badge LIVE
                cv2.rectangle(annotated_img, (10, 80), (100, 110), (0, 0, 255), -1)
                cv2.putText(annotated_img, "LIVE", 
                           (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                return frame.from_ndarray(annotated_img, format="bgr24")
                
            except Exception as e:
                # En cas d'erreur, afficher l'image originale
                cv2.putText(img, f"Error: {str(e)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                return frame.from_ndarray(img, format="bgr24")

# ========== CHARGEMENT AUTOMATIQUE DU MODÈLE ==========
if 'model' not in st.session_state or st.session_state.model is None:
    with st.spinner("🔄 Chargement du modèle YOLOv8..."):
        st.session_state.model = download_and_load_model()
        if st.session_state.model is not None:
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False

# Titre principal
st.markdown('<p class="main-header">🗑️ Détecteur de Poubelles YOLOv8</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Détection en temps réel avec caméra</p>', unsafe_allow_html=True)

# Message de statut du modèle
if st.session_state.model_loaded:
    st.success("✅ Modèle chargé et prêt à l'emploi!")
else:
    st.error("❌ Impossible de charger le modèle. Veuillez vérifier la configuration.")
    st.stop()

# Sidebar - Configuration
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
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            st.download_button(
                label="⬇️ Télécharger best.pt",
                data=file,
                file_name="best.pt",
                mime="application/octet-stream",
                use_container_width=True
            )
    
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
tab1, tab2, tab3, tab4 = st.tabs(["📹 Caméra Temps Réel", "📸 Images", "🎥 Vidéos", "ℹ️ À Propos"])

# ========== TAB 1: WEBCAM TEMPS RÉEL ==========
with tab1:
    st.header("📹 Détection en Temps Réel")
    
    st.markdown("""
    <div class="success-box">
        <h3>🎥 Webcam en Direct</h3>
        <p>Activez votre webcam pour une détection en temps réel!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📷 Flux Vidéo")
        
        # Créer le contexte WebRTC
        webrtc_ctx = webrtc_streamer(
            key="yolo-detection",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Configurer le modèle dans le processor
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_model(st.session_state.model)
            webrtc_ctx.video_processor.set_confidence(confidence)
            webrtc_ctx.video_processor.set_iou(iou_threshold)
    
    with col2:
        st.markdown("### 📊 Statistiques")
        
        if webrtc_ctx.video_processor:
            stats_placeholder = st.empty()
            
            # Afficher les statistiques en temps réel
            if webrtc_ctx.state.playing:
                st.markdown('<p class="live-badge">🔴 EN DIRECT</p>', unsafe_allow_html=True)
                
                with stats_placeholder.container():
                    processor = webrtc_ctx.video_processor
                    
                    st.metric("Frames traités", processor.frame_count)
                    st.metric("Détections totales", processor.detection_count)
                    
                    if processor.frame_count > 0:
                        avg_detections = processor.detection_count / processor.frame_count
                        st.metric("Moyenne/frame", f"{avg_detections:.2f}")
                    
                    if len(processor.fps_list) > 0:
                        st.metric("FPS moyen", f"{np.mean(processor.fps_list):.1f}")
            else:
                st.info("▶️ Cliquez sur START pour commencer")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Contrôles")
        st.info("""
        **Instructions:**
        1. Cliquez sur **START**
        2. Autorisez l'accès à la caméra
        3. Positionnez la poubelle devant la caméra
        4. Les détections s'affichent en temps réel!
        
        **Arrêter:** Cliquez sur **STOP**
        """)
    
    st.markdown("---")
    
    # Informations supplémentaires
    with st.expander("💡 Conseils pour une meilleure détection"):
        st.markdown("""
        ### 📌 Pour de meilleurs résultats:
        
        1. **Éclairage** 💡
           - Assurez un bon éclairage
           - Évitez les contre-jours
        
        2. **Distance** 📏
           - Gardez la poubelle à 1-3 mètres
           - Cadrez entièrement l'objet
        
        3. **Stabilité** 🎯
           - Gardez la caméra stable
           - Évitez les mouvements brusques
        
        4. **Angle** 📐
           - Vue frontale ou légèrement en hauteur
           - Évitez les angles trop obliques
        
        5. **Paramètres** ⚙️
           - Ajustez la confiance si trop/pas assez de détections
           - Réduisez l'IoU si détections dupliquées
        """)

# ========== TAB 2: DETECTION SUR IMAGES ==========
with tab2:
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
                    img_array = np.array(image)
                    
                    results = st.session_state.model.predict(
                        img_array,
                        conf=confidence,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    annotated = results[0].plot(
                        line_width=box_thickness,
                        labels=show_labels,
                        conf=show_conf
                    )
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_rgb, caption="Détections", use_container_width=True)
                    
                    boxes = results[0].boxes
                    num_detections = len(boxes)
                    
                    st.markdown(f"""
                    <div class="stat-box">
                        <h2>{num_detections}</h2>
                        <p>Objets détectés</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if num_detections > 0:
                        with st.expander("📊 Détails des détections"):
                            for i, box in enumerate(boxes):
                                cls = int(box.cls[0])
                                conf_val = float(box.conf[0])
                                label = st.session_state.model.names[cls]
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                st.write(f"**Objet {i+1}:**")
                                st.write(f"  - Classe: {label}")
                                st.write(f"  - Confiance: {conf_val:.2%}")
                                st.write(f"  - Position: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
                                st.markdown("---")
                        
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

# ========== TAB 3: DETECTION SUR VIDEOS ==========
with tab3:
    st.header("🎥 Détection sur Vidéos")
    
    uploaded_video = st.file_uploader(
        "Choisissez une vidéo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
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
                    cap = cv2.VideoCapture(video_path)
                    
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"📹 Vidéo: {width}x{height} @ {fps} FPS - {total_frames} frames")
                    
                    output_path = tempfile.mktemp(suffix='.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
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
                        
                        results = st.session_state.model.predict(
                            frame,
                            conf=confidence,
                            iou=iou_threshold,
                            verbose=False
                        )
                        
                        annotated_frame = results[0].plot(
                            line_width=box_thickness,
                            labels=show_labels,
                            conf=show_conf
                        )
                        
                        out.write(annotated_frame)
                        
                        num_detections = len(results[0].boxes)
                        detection_stats.append(num_detections)
                        
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
                            
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    cap.release()
                    out.release()
                    
                    progress_bar.progress(1.0)
                    processing_time = time.time() - start_time
                    status_text.text(f"✅ Traitement terminé en {processing_time:.1f}s!")
                    
                    st.success("🎉 Détection terminée!")
                    
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
                    
                    try:
                        cap.release()
                        time.sleep(0.5)
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    if 'cap' in locals():
                        cap.release()
                    try:
                        if 'video_path' in locals() and os.path.exists(video_path):
                            os.unlink(video_path)
                    except:
                        pass
    else:
        st.info("👆 Uploadez une vidéo pour commencer")

# ========== TAB 4: À PROPOS ==========
with tab4:
    st.header("ℹ️ À Propos de l'Application")
    
    st.markdown("""
    ### 🎯 Objectif
    Cette application utilise l'intelligence artificielle (YOLOv8) pour détecter automatiquement 
    si les poubelles sont **pleines** ou **vides**.
    
    ### 🚀 Fonctionnalités
    - ✅ **Détection en temps réel via webcam** (NOUVEAU!)
    - ✅ Détection instantanée sur images
    - ✅ Traitement de vidéos complètes
    - ✅ Téléchargement des résultats annotés
    - ✅ Téléchargement du modèle pour utilisation hors ligne
    - ✅ Paramètres ajustables en temps réel
    
    ### 🛠️ Technologies Utilisées
    - **YOLOv8** (Ultralytics) - Détection d'objets
    - **Streamlit** - Interface web
    - **Streamlit-WebRTC** - Streaming vidéo en temps réel
    - **OpenCV** - Traitement vidéo
    - **Python** - Langage de programmation
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #666;'>
        <p><strong>🗑️ Détecteur de Poubelles YOLOv8</strong></p>
        <p>Développé avec ❤️ using Streamlit & Ultralytics YOLO</p>
        <p><small>Version 2.0 avec WebRTC - 2024</small></p>
    </div>
    """, unsafe_allow_html=True)
