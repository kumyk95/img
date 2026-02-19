import cv2
import numpy as np
import time
import os
import dlib
import glob
try:
    import mediapipe as mp
except ImportError:
    mp = None
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None
import torch

class AIService:
    def __init__(self):
        # Load models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize Dlib
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "/root/site-img/backend/static/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print(f"Dlib predictor loaded from {predictor_path}")
        else:
            self.predictor = None
            print(f"WARNING: Dlib predictor not found at {predictor_path}")
            
        # Initialize PCA Model
        self.pca_mean = None
        self.pca_eigenvectors = None
        self.images_processed = 0  # Counter for auto-retraining
        
        # Initialize Mediapipe
        if mp and hasattr(mp, 'solutions'):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except Exception:
                self.mp_face_mesh = None
        else:
            self.mp_face_mesh = None
        
        # Initialize GFPGAN for CNN method
        self.gfpgan_restorer = None
        gfpgan_model_path = "/root/site-img/backend/models/GFPGANv1.4.pth"
        
        # Initialize CodeFormer for GAN method
        self.codeformer_net = None
        codeformer_model_path = "/root/site-img/backend/models/codeformer.pth"
        self.device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'

        if GFPGANer and os.path.exists(gfpgan_model_path):
            try:
                self.gfpgan_restorer = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.available_device()
                )
                print(f"GFPGAN model loaded on {self.device}")
            except Exception as e:
                print(f"WARNING: GFPGAN init failed: {e}")
                self.gfpgan_restorer = None

        if os.path.exists(codeformer_model_path):
            try:
                from codeformer.basicsr.archs.codeformer_arch import CodeFormer
                self.codeformer_net = CodeFormer(
                    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                    connect_list=['32', '64', '128', '256']
                ).to(self.device)
                ckpt = torch.load(codeformer_model_path, map_location=self.device)['params_ema']
                self.codeformer_net.load_state_dict(ckpt)
                self.codeformer_net.eval()
                print("CodeFormer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize CodeFormer: {e}")
                
        self.train_pca_model()

    def available_device(self):
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_codeformer(self, img_input):
        """Helper to run CodeFormer inference"""
        if self.codeformer_net is None: return img_input
        import torch
        h, w = img_input.shape[:2]
        img_resized = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.codeformer_net(img_t, w=0.5, adain=True)[0]
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.
        output = output.astype('uint8')
        if (h, w) != (512, 512):
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return output



    def train_pca_model(self):
        """Train a simple Eigenfaces model using available original images."""
        log_file = "/root/site-img/backend/pca_train.log"
        with open(log_file, "a") as f_log:  # Changed to append mode
            try:
                f_log.write(f"\n{time.ctime()}: Training PCA model started...\n")
                images = []
                static_dir = "/root/site-img/backend/static"
                files = glob.glob(os.path.join(static_dir, "original_*.jpg"))
                f_log.write(f"Found {len(files)} candidate files\n")
                
                for path in files:
                    img = cv2.imread(path)
                    if img is None: 
                        continue
                    
                    # Detect face to normalize
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Try Cascade with more lenient parameters
                    faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
                    found_face = False
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        found_face = True
                    else:
                        # Try Dlib
                        rects = self.detector(gray, 0)
                        if len(rects) > 0:
                            x, y, w, h = rects[0].left(), rects[0].top(), rects[0].width(), rects[0].height()
                            x, y = max(0, x), max(0, y)
                            w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
                            if w > 10 and h > 10:
                                found_face = True
                    
                    if found_face:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (128, 128))
                        images.append(face.flatten().astype(np.float32))
                
                f_log.write(f"Faces successfully extracted: {len(images)}\n")
                if len(images) > 3:
                    data_matrix = np.array(images)
                    # Increased from 50 to 100 components for better quality
                    max_comp = min(len(images) - 1, 100)
                    mean, eigenvectors = cv2.PCACompute(data_matrix, mean=None, maxComponents=max_comp)
                    self.pca_mean = mean
                    self.pca_eigenvectors = eigenvectors
                    f_log.write(f"PCA Model trained successfully on {len(images)} faces with {len(eigenvectors)} components\n")
                else:
                    f_log.write(f"Not enough faces found for PCA training (found {len(images)})\n")
            except Exception as e:
                f_log.write(f"PCA training failed: {e}\n")

    def recognize_face(self, current_image_path, original_image_path):
        """Compare current image with original image to get similarity score."""
        start_time = time.time()
        try:
            from deepface import DeepFace
            # Using VGG-Face which is robust
            result = DeepFace.verify(
                img1_path=original_image_path,
                img2_path=current_image_path,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="opencv",
                distance_metric="cosine"
            )
            distance = result.get('distance', 1.0)
            score = max(0, 1.0 - distance)
            recognition_time = (time.time() - start_time) * 1000
            return score, recognition_time
        except Exception as e:
            print(f"DeepFace error: {e}")
            return 0.0, 0.0

    def get_landmarks(self, img):
        """Helper to get 68 landmarks using Dlib."""
        if self.predictor is None: 
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) == 0: 
            return None
        
        shape = self.predictor(gray, rects[0])
        landmarks = []
        for i in range(0, 68):
            landmarks.append((shape.part(i).x, shape.part(i).y))
        return landmarks

    def get_mesh_landmarks(self, img):
        """Get 468 landmarks using Mediapipe."""
        if not self.mp_face_mesh:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        
        h, w = img.shape[:2]
        mesh_landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            mesh_landmarks.append((int(lm.x * w), int(lm.y * h)))
        return mesh_landmarks

    def get_eye_region_mask(self, img, landmarks, dilate_amount=3):
        """Create a precise mask covering ONLY the glasses/eye region."""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        mesh_landmarks = self.get_mesh_landmarks(img)
        
        if mesh_landmarks:
            # Use Mediapipe 468 points for extreme precision
            # Left eye contours
            left_eye = [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263]
            # Right eye contours
            right_eye = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
            # Bridge
            bridge = [168, 6, 197, 195, 5]
            
            pts_l = np.array([mesh_landmarks[i] for i in left_eye], np.int32)
            pts_r = np.array([mesh_landmarks[i] for i in right_eye], np.int32)
            
            # Draw eyes with slight expansion to cover frames
            cv2.fillPoly(mask, [pts_l], 255)
            cv2.fillPoly(mask, [pts_r], 255)
            
            # Bridge area
            for i in range(len(bridge)-1):
                cv2.line(mask, mesh_landmarks[bridge[i]], mesh_landmarks[bridge[i+1]], 255, thickness=int(w*0.02))
        
        elif landmarks:
            # Dlib 68 points — build a tight mask per eye using eye + eyebrow
            # Left eye region: eyebrow top (17-21) + eye bottom (36-41)
            left_region = []
            for i in [17, 18, 19, 20, 21]:  # left eyebrow (top boundary)
                left_region.append(landmarks[i])
            for i in [39, 40, 41, 36]:  # left eye bottom (lower boundary)
                left_region.append(landmarks[i])
            
            # Right eye region: eyebrow top (22-26) + eye bottom (42-47)
            right_region = []
            for i in [22, 23, 24, 25, 26]:  # right eyebrow (top boundary)
                right_region.append(landmarks[i])
            for i in [45, 46, 47, 42]:  # right eye bottom (lower boundary)
                right_region.append(landmarks[i])
            
            pts_left = np.array(left_region, np.int32)
            pts_right = np.array(right_region, np.int32)
            cv2.fillPoly(mask, [cv2.convexHull(pts_left)], 255)
            cv2.fillPoly(mask, [cv2.convexHull(pts_right)], 255)
            
            # Nose bridge — narrow line only between the eyes
            cv2.line(mask, landmarks[39], landmarks[42], 255, thickness=int(w * 0.015))
        
        else:
            # Last resort fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                cv2.rectangle(mask, (x + int(fw*0.1), y + int(fh*0.28)), 
                             (x + int(fw*0.9), y + int(fh*0.42)), 255, -1)
        
        # Reduced dilation for "minimal impact" on healthy skin
        # iterations=1 or 2 instead of 3
        mask = cv2.dilate(mask, np.ones((dilate_amount, dilate_amount), np.uint8), iterations=1)
        return mask

    def add_sunglasses(self, image_path, glasses_path=None):
        start_time = time.time()
        img = cv2.imread(image_path)
        if img is None: 
            raise ValueError(f"Image not found at {image_path}")
            
        if glasses_path is None:
            candidates = [
                "/root/site-img/public/assets/glasses.png",
                "/root/site-img/backend/static/glasses.png"
            ]
            for c in candidates:
                if os.path.exists(c):
                    glasses_path = c
                    break

        landmarks = self.get_landmarks(img)
        processed = False
        
        if landmarks and glasses_path and os.path.exists(glasses_path):
            try:
                l_eye = np.mean([landmarks[36], landmarks[39]], axis=0).astype(int)
                r_eye = np.mean([landmarks[42], landmarks[45]], axis=0).astype(int)
                dY, dX = r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]
                angle = np.degrees(np.arctan2(dY, dX))
                center = ((l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2)
                eye_dist = np.linalg.norm(np.array(landmarks[45]) - np.array(landmarks[36]))
                glasses_width = int(eye_dist * 2.3)
                
                glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
                if glasses is not None:
                    ratio = glasses_width / glasses.shape[1]
                    glasses_height = int(glasses.shape[0] * ratio)
                    resized = cv2.resize(glasses, (glasses_width, glasses_height))
                    M = cv2.getRotationMatrix2D((glasses_width//2, glasses_height//2), -angle, 1.0)
                    rotated = cv2.warpAffine(resized, M, (glasses_width, glasses_height), 
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                    self.overlay_image_alpha(img, rotated, center[0] - glasses_width//2, center[1] - glasses_height//2)
                    processed = True
            except Exception as e:
                print(f"Error adding glasses: {e}")

        if not processed:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.circle(img, (x + int(w*0.3), y + int(h*0.35)), int(w*0.12), (20, 20, 20), -1)
                cv2.circle(img, (x + int(w*0.7), y + int(h*0.35)), int(w*0.12), (20, 20, 20), -1)
                cv2.line(img, (x + int(w*0.3), y + int(h*0.35)), (x + int(w*0.7), y + int(h*0.35)), (20, 20, 20), 4)

        process_time = (time.time() - start_time) * 1000
        return img, process_time

    def overlay_image_alpha(self, img, img_overlay, x, y):
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return
        overlay_part = img_overlay[y1o:y2o, x1o:x2o]
        if overlay_part.shape[2] == 4:
            alpha = overlay_part[:, :, 3:4] / 255.0
            img[y1:y2, x1:x2] = (alpha * overlay_part[:, :, :3] + (1 - alpha) * img[y1:y2, x1:x2]).astype(np.uint8)
        else:
            img[y1:y2, x1:x2] = overlay_part

    def blend_result(self, original, restored, mask):
        """Seamlessly blend restored eye region back into the original face."""
        if original.shape != restored.shape:
            restored = cv2.resize(restored, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            
        # Create a smooth feathered mask around the eyes
        h, w = original.shape[:2]
        dilation_size = max(int(min(h, w) * 0.05), 5)
        # Dilate mask to cover the whole glasses area
        wide_mask = cv2.dilate(mask, np.ones((dilation_size, dilation_size), np.uint8), iterations=2)
        
        # Soften edges
        blur_size = max(int(min(h, w) * 0.08) | 1, 11)
        mask_float = wide_mask.astype(np.float32) / 255.0
        mask_smooth = cv2.GaussianBlur(mask_float, (blur_size, blur_size), blur_size // 3)
        mask_3c = np.stack([mask_smooth] * 3, axis=-1)
        
        # Blend
        blended = (restored.astype(np.float32) * mask_3c + 
                   original.astype(np.float32) * (1.0 - mask_3c))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def process_reconstruction(self, image_path, original_path, method_id):
        # Auto-retrain PCA model occasionally
        self.images_processed += 1
        if self.images_processed % 50 == 0:
            self.train_pca_model()
            
        start_time = time.time()
        img_with_glasses = cv2.imread(image_path) # The one with black circles
        img_original = cv2.imread(original_path)  # The clean one (target/reference)
        
        if img_with_glasses is None or img_original is None:
            raise ValueError("Could not load images")

        # Detect landmarks first (REQUIRED for get_eye_region_mask)
        landmarks = self.get_landmarks(img_original)
        if landmarks is None:
            landmarks = self.get_landmarks(img_with_glasses)

        # Get precise mask using landmarks
        mask = self.get_eye_region_mask(img_with_glasses, landmarks, dilate_amount=5)
        
        # Prepare the input for AI by removing the black holes first
        # We use a neutral skin-color sample from the original image if possible
        prepared = cv2.inpaint(img_with_glasses, mask, 5, cv2.INPAINT_TELEA)

        result = prepared.copy()

        if method_id == 3:
            # Step 3: Pure Inpainting (Baseline)
            result = prepared
            
        elif method_id == 4:
            # Step 4: CNN (GFPGAN)
            if self.gfpgan_restorer:
                try:
                    _, _, restored_face = self.gfpgan_restorer.enhance(
                        prepared, has_aligned=False, only_center_face=True, paste_back=True
                    )
                    result = self.blend_result(img_original, restored_face, mask)
                except Exception as e:
                    print(f"GFPGAN failed: {e}")
                    result = prepared
            else: result = prepared
                
        elif method_id == 5:
            # Step 5: GAN (CodeFormer)
            if self.codeformer_net:
                try:
                    restored_face = self.run_codeformer(prepared)
                    result = self.blend_result(img_original, restored_face, mask)
                except Exception as e:
                    print(f"CodeFormer failed: {e}")
                    result = prepared
            else: result = prepared
                
        elif method_id == 6:
            # Step 6: Hybrid (Ensemble)
            res_gfpgan = prepared
            res_codeformer = prepared
            
            if self.gfpgan_restorer:
                try:
                    _, _, out = self.gfpgan_restorer.enhance(prepared, has_aligned=False, only_center_face=True, paste_back=True)
                    res_gfpgan = out
                except: pass
            if self.codeformer_net:
                try: res_codeformer = self.run_codeformer(prepared)
                except: pass
            
            # Weighted average of restorations
            try:
                restored_ensemble = cv2.addWeighted(res_gfpgan.astype(np.float32), 0.5, 
                                                   res_codeformer.astype(np.float32), 0.5, 0).astype(np.uint8)
                result = self.blend_result(img_original, restored_ensemble, mask)
            except:
                result = res_gfpgan

        else:
            result = prepared

        process_time = (time.time() - start_time) * 1000
        return result, process_time
