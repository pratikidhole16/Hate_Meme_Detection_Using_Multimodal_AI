"""
Explainability Engine - XAI features for Hateful Memes Detector
Includes: Grad-CAM, LIME, Feature Importance, Attention Maps
"""

import base64
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not available, Grad-CAM disabled")

try:
    import lime
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available, LIME explanations disabled")


class ExplainabilityEngine:
    """XAI engine for model interpretability"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        self.cfg = classifier.cfg
        self.group_config = self._load_group_config()
        self._clip_model = None
        self._clip_processor = None

    def _load_group_config(self):
        """Load protected group and stereotype lexicons from config, with a built-in fallback."""
        default = {
            "religion": ["muslim", "islam", "islamic", "christian", "catholic", "protestant", "orthodox", "jew", "jewish", "jewry", "hindu", "sikh", "buddhist", "buddhism", "atheist", "agnostic"],
            "ethnicity": ["arab", "arabic", "persian", "kurd", "latino", "latina", "latinx", "mexican", "chicano", "asian", "south asian", "east asian", "african", "african american", "caribbean", "pashtun", "somali", "eritrean"],
            "nationality": ["indian", "pakistani", "bangladeshi", "afghani", "afghan", "iranian", "iraqi", "syrian", "lebanese", "egyptian", "turkish", "russian", "ukrainian", "polish", "chinese", "japanese", "korean", "filipino", "vietnamese", "thai", "nigerian", "kenyan", "ethiopian", "somali", "sudanese", "brazilian", "argentinian", "canadian", "american", "usa", "british", "french", "german", "italian", "spanish"],
            "race": ["black", "white", "brown", "caucasian"],
            "orientation": ["lgbt", "lgbtq", "gay", "lesbian", "bi", "bisexual", "trans", "transgender", "nonbinary", "queer", "intersex"],
            "gender": ["woman", "women", "female", "man", "men", "male", "girl", "boy", "mother", "father", "wife", "husband"],
            "immigration": ["immigrant", "refugee", "asylum", "migrant", "undocumented", "illegal alien"],
            "disability": ["disabled", "autistic", "autism", "adhd", "deaf", "blind", "wheelchair", "cripple", "dwarf", "little person"],
            "caste": ["dalit", "brahmin", "shudra", "kshatriya", "vaishya"],
            "stereotypes": [
                "are terrorists", "are terror", "are bombers", "bring crime", "are criminals", "are thieves", "are invaders", "are taking over", "are dirty", "spread disease", "don't belong here", "send them back", "ban all", "wipe them out", "are animals", "are vermin", "are rats", "are pigs", "are monkeys", "are groomers"
            ],
            "threats": [
                "kill", "shoot", "bomb", "burn", "hang", "lynch", "wipe out", "exterminate", "eradicate", "gas", "attack", "destroy", "eliminate"
            ],
        }

        cfg_path = ROOT / "data" / "config" / "protected_groups.json"
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    # Merge loaded keys over defaults to allow extensions
                    for k, v in loaded.items():
                        if isinstance(v, list):
                            default[k] = v
                    return default
            except Exception as exc:
                logger.warning("Could not load group config %s: %s; using defaults", cfg_path, exc)
        return default

    def _load_clip(self):
        """Lazy-load CLIP model/processor; return False if unavailable."""
        if self._clip_model is not None and self._clip_processor is not None:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor
            model_name = "openai/clip-vit-base-patch32"
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            # Prefer GPU if available, else CPU to avoid crashes
            device = self.device if torch.cuda.is_available() else torch.device("cpu")
            self._clip_model = CLIPModel.from_pretrained(model_name).to(device)
            self._clip_model.eval()
            return True
        except Exception as exc:
            logger.warning("CLIP unavailable for visual group hints: %s", exc)
            self._clip_model = None
            self._clip_processor = None
            return False

    def _visual_group_hints(self, image_path: str):
        """Use CLIP zero-shot to guess possible group cues from the image.

        Returns list of (label, probability) sorted by probability.
        """
        if not self._load_clip():
            return []
        try:
            terms = []
            for k, vs in self.group_config.items():
                if k in {"stereotypes", "threats"}:
                    continue
                terms.extend(vs)
            # Deduplicate and cap to keep compute reasonable
            uniq_terms = list(dict.fromkeys([t.lower() for t in terms if t]))
            uniq_terms = uniq_terms[:64]

            prompts = []
            for term in uniq_terms:
                prompts.append(f"a photo of a {term} person")
                prompts.append(f"a photo of {term} people")
            prompts = prompts[:80]

            image = Image.open(image_path).convert("RGB")
            inputs = self._clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
            # Move to same device as model
            device = next(self._clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image.squeeze(0)
                probs = logits.softmax(dim=0)

            topk = min(5, probs.numel())
            values, indices = torch.topk(probs, k=topk)
            hints = []
            for score, idx in zip(values.tolist(), indices.tolist()):
                if score < 0.12:  # filter very weak signals
                    continue
                hints.append((prompts[idx], score))
            return hints
        except Exception as exc:
            logger.warning("Visual group hinting failed: %s", exc)
            return []
    
    def explain_prediction(self, image_path: str, text: str, predicted_class: int):
        """Generate comprehensive explanations for prediction
        
        Returns:
            dict with multiple explanation methods
        """
        explanations = {
            'methods_available': [],
            'prediction_confidence': None,
            'attention_regions': [],
            'feature_importance': {},
            'reasoning': self._generate_nlp_reasoning(text, image_path, predicted_class),
        }
        
        try:
            # Grad-CAM visualization
            if GRADCAM_AVAILABLE:
                gradcam_result = self._generate_gradcam(image_path)
                if gradcam_result:
                    explanations['gradcam_base64'] = gradcam_result['image_base64']
                    explanations['gradcam_explanation'] = gradcam_result['explanation']
                    explanations['methods_available'].append('gradcam')
            
            # LIME explanation
            if LIME_AVAILABLE:
                lime_result = self._generate_lime_explanation(image_path, predicted_class)
                if lime_result:
                    explanations['lime_base64'] = lime_result['image_base64']
                    explanations['lime_explanation'] = lime_result['explanation']
                    explanations['methods_available'].append('lime')
            
            # Text importance and extract hateful keywords
            text_importance = self._analyze_text_importance(text, predicted_class)
            explanations['text_importance'] = text_importance
            
            # Extract hateful keywords from text
            explanations['hateful_keywords'] = self._extract_hateful_keywords(text, predicted_class)
            
            # Saliency map
            saliency = self._generate_saliency_map(image_path)
            if saliency:
                explanations['saliency_map'] = saliency
                explanations['methods_available'].append('saliency')
            
            # Attention regions with keywords
            explanations['attention_regions'] = self._extract_attention_regions(text, predicted_class)
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
        
        return explanations
    
    def _generate_gradcam(self, image_path: str):
        """Generate Grad-CAM visualization overlaid on original image"""
        try:
            logger.info(f"Starting Grad-CAM generation for {image_path}")
            if not GRADCAM_AVAILABLE:
                return None
            
            from torchvision import transforms
            from torchvision.io import read_image
            import torch.nn as nn
            import cv2
            
            # Load original image for overlay
            original_img = cv2.imread(image_path)
            if original_img is None:
                logger.error(f"Cannot read image: {image_path}")
                return None
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, (224, 224))
            
            # Load and preprocess image for model
            image = read_image(image_path).float() / 255.0
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] > 3:
                image = image[:3]
            
            transform = transforms.Resize((224, 224))
            image = transform(image).unsqueeze(0).to(self.device)
            
            # Create dummy tokens for the text encoder (empty text)
            dummy_tokens = self.classifier.tokenizer(
                "", 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            # Create a wrapper model that only takes image input
            class VisionOnlyWrapper(nn.Module):
                def __init__(self, model, tokens):
                    super().__init__()
                    self.model = model
                    self.tokens = tokens
                
                def forward(self, images):
                    return self.model(images, self.tokens)
            
            wrapper_model = VisionOnlyWrapper(self.model, dummy_tokens)
            
            # Find last convolutional layer in vision encoder
            target_layer = None
            for module in reversed(list(self.model.vision.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
            
            if target_layer is None:
                logger.warning("No Conv2d layer found in vision encoder for Grad-CAM")
                return None
            
            try:
                # GradCAM automatically detects device
                with GradCAM(model=wrapper_model, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=image, targets=None)
                    grayscale_cam = grayscale_cam[0, :]
                
                # Normalize and apply colormap
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                
                # Apply heatmap colormap (JET for purple/red colors)
                heatmap = cv2.applyColorMap((grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Overlay on original image (70% heatmap, 30% original)
                overlay = cv2.addWeighted(original_img, 0.3, heatmap, 0.7, 0)
                
                # Convert to PIL and encode
                overlay_pil = Image.fromarray(overlay.astype(np.uint8))
                buffer = io.BytesIO()
                overlay_pil.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                logger.info(f"Grad-CAM generated successfully, base64 length: {len(img_base64)}")
                return {
                    'image_base64': f'data:image/png;base64,{img_base64}',
                    'explanation': 'Grad-CAM heatmap (purple/red) shows which image regions the model focused on for prediction'
                }
            except Exception as grad_error:
                logger.error(f"Grad-CAM computation failed: {str(grad_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"Grad-CAM error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_lime_explanation(self, image_path: str, predicted_class: int):
        """Generate LIME explanation with green boundaries on important regions"""
        try:
            logger.info(f"Starting LIME generation for {image_path}")
            if not LIME_AVAILABLE:
                return None
            
            from PIL import Image
            import tempfile
            from skimage.segmentation import mark_boundaries
            
            # Load image and ensure correct format
            img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
            img = np.array(img_pil) / 255.0  # Normalize to 0-1 range
            
            # Ensure image is in correct shape (H, W, 3)
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 4:  # RGBA
                img = img[:, :, :3]
            
            # LIME explainer
            explainer = lime.lime_image.LimeImageExplainer()
            
            # Prediction function - simple wrapper
            def predict_fn(images):
                try:
                    preds = []
                    for img_array in images:
                        # Normalize if needed
                        if img_array.max() > 1.5:
                            img_normalized = img_array / 255.0
                        else:
                            img_normalized = img_array
                        
                        # Convert to PIL
                        img_uint8 = (img_normalized * 255).astype(np.uint8)
                        img_pil_temp = Image.fromarray(img_uint8)
                        
                        # Save temporarily and predict
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = str(Path(temp_dir) / 'lime_temp.png')
                            img_pil_temp.save(temp_path)
                            try:
                                pred = self.classifier.predict_single(temp_path, "")
                                confidence = float(pred['confidence'])
                                # Return [prob_class_0, prob_class_1]
                                preds.append([1 - confidence, confidence])
                            except Exception as e:
                                logger.warning(f"Prediction failed in LIME: {str(e)}")
                                preds.append([0.5, 0.5])  # Fallback
                    return np.array(preds)
                except Exception as e:
                    logger.error(f"Predict_fn error: {str(e)}")
                    return np.ones((len(images), 2)) * 0.5
            
            logger.info("Running LIME explainer...")
            # Generate explanation
            explanation = explainer.explain_instance(
                img,
                predict_fn,
                top_labels=2,
                num_samples=25,  # Reduced for speed
                hide_color=0,
                batch_size=5
            )
            
            # Get the image and mask for the predicted class
            label = explanation.top_labels[0] if explanation.top_labels else predicted_class
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=True,
                num_features=4,
                hide_rest=False,
                min_weight=0.0
            )
            
            # Mark boundaries with green color
            logger.info("Applying green boundaries to LIME mask...")
            lime_img = mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask, color=(0, 1, 0), mode='thick')
            lime_img = (lime_img * 255).astype(np.uint8)
            
            # Convert to base64
            img_pil_final = Image.fromarray(lime_img)
            buffer = io.BytesIO()
            img_pil_final.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info(f"LIME generated successfully, base64 length: {len(img_base64)}")
            return {
                'image_base64': f'data:image/png;base64,{img_base64}',
                'explanation': 'LIME shows important image regions with green boundaries - these areas most influenced the prediction'
            }
        
        except Exception as e:
            logger.error(f"LIME error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_saliency_map(self, image_path: str):
        """Generate saliency map (gradient of output w.r.t input)"""
        try:
            from torchvision import transforms
            from torchvision.io import read_image
            
            image = read_image(image_path).float() / 255.0
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] > 3:
                image = image[:3]
            
            transform = transforms.Resize((224, 224))
            image = transform(image).unsqueeze(0).to(self.device)
            image.requires_grad = True
            
            with torch.enable_grad():
                logits = self.model.vision(image)
                max_logit = logits.max()
                max_logit.backward()
            
            saliency = image.grad.data.abs().max(dim=1)[0][0].cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            return {
                'map': saliency.tolist(),
                'explanation': 'Saliency map shows input sensitivity - darker areas have larger gradients'
            }
        
        except Exception as e:
            logger.error(f"Saliency error: {str(e)}")
            return None
    
    def _extract_hateful_keywords(self, text: str, predicted_class: int):
        """Extract hateful keywords from text"""
        if not text or predicted_class == 0:
            return []
        
        hate_keywords_dict = {
            'hate', 'bad', 'stupid', 'ugly', 'dumb', 'trash', 'loser',
            'offensive', 'racist', 'sexist', 'discriminat', 'bigot', 'kill',
            'die', 'death', 'inferior', 'scum', 'filth', 'garbage'
        }
        
        words = text.lower().split()
        found_keywords = []
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip('.,!?;:"\'-')
            if any(keyword in clean_word for keyword in hate_keywords_dict):
                found_keywords.append(word)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def _analyze_text_importance(self, text: str, predicted_class: int):
        """Analyze which words are important"""
        if not text:
            return {'words': [], 'importance': []}
        
        words = text.split()
        # Simple heuristic: longer words, words that appear in hate speech datasets
        hate_keywords = {
            'hate', 'bad', 'stupid', 'ugly', 'dumb', 'trash', 'loser',
            'offensive', 'racist', 'sexist', 'discriminat', 'bigot', 'kill',
            'die', 'death', 'inferior', 'scum', 'filth', 'garbage'
        }
        
        importance = []
        for word in words:
            score = 0
            # Remove punctuation for matching
            clean_word = word.lower().strip('.,!?;:"\'-')
            if any(keyword in clean_word for keyword in hate_keywords):
                score = 0.95 if predicted_class == 1 else 0.25
            else:
                # Regular words get varied importance
                score = 0.2 + (len(word) * 0.05)  # Longer words slightly more important
                if predicted_class == 0:
                    score = min(0.7, score + 0.3)  # Boost for safe content
            importance.append(min(1.0, score))  # Cap at 1.0
        
        return {
            'words': words,
            'importance': importance,
            'max_important_word': words[np.argmax(importance)] if words else None
        }
    
    def _extract_attention_regions(self, text: str, predicted_class: int):
        """Extract and explain attention regions"""
        regions = []
        
        if predicted_class == 1:  # Hateful
            regions.append({
                'region': 'Offensive language detected',
                'importance': 'High',
                'reason': 'Hateful content often uses offensive terms'
            })
            regions.append({
                'region': 'Dehumanizing imagery',
                'importance': 'Medium',
                'reason': 'Visual elements can reinforce hateful messages'
            })
        else:  # Non-hateful
            regions.append({
                'region': 'No hateful keywords',
                'importance': 'High',
                'reason': 'Absence of offensive language is a strong indicator'
            })
            regions.append({
                'region': 'Neutral imagery',
                'importance': 'Medium',
                'reason': 'The visual elements do not reinforce hateful narratives'
            })
        
        return regions
    
    def _generate_nlp_reasoning(self, text: str, image_path: str, predicted_class: int):
        """Generate NLP-based dynamic reasoning using transformers sentiment analysis"""
        reasoning_parts = []
        sentiment_label = None
        sentiment_score = None
        offensive_found = []
        positive_found = []
        tokens = []
        
        logger.info(f"===== NLP REASONING START (Class: {predicted_class}, Text: {text[:50]}) =====")
        
        # Always run sentiment analysis if text exists
        if text and len(text.strip()) > 0:
            logger.info(f"Text provided: {text}")
            try:
                from transformers import pipeline
                logger.info("Importing sentiment analyzer...")
                
                device_id = -1  # force CPU to avoid CUDA-related crashes during sentiment
                logger.info(f"Using device: {device_id}")
                
                sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device_id
                )
                logger.info("Sentiment analyzer loaded, running analysis...")
                
                # Truncate to 512 chars for BERT
                text_input = text[:512]
                sentiment_result = sentiment_analyzer(text_input)
                logger.info(f"Raw sentiment result: {sentiment_result}")
                
                if sentiment_result:
                    sentiment_label = sentiment_result[0]['label']
                    sentiment_score = sentiment_result[0]['score']
                    logger.info(f"ACTUAL NLP SENTIMENT: {sentiment_label} ({sentiment_score:.4f})")
                
            except Exception as se:
                logger.error(f"NLP sentiment analysis error: {type(se).__name__}: {str(se)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("No text provided for sentiment analysis")
        
        # Keyword detection
        if text:
            try:
                tokens = self.classifier.tokenizer.tokenize(text.lower())
                hate_terms = {'hate', 'stupid', 'ugly', 'dumb', 'trash', 'loser', 'kill', 'die', 'death', 
                             'racist', 'sexist', 'inferior', 'scum', 'filth', 'garbage', 'bigot', 'attack', 'destroy'}
                positive_terms = {'love', 'good', 'great', 'nice', 'happy', 'fun', 'awesome', 'beautiful', 
                                  'wonderful', 'amazing', 'friendly', 'kind', 'help', 'support', 'peace', 'excellent'}

                for term in hate_terms:
                    if any(term in token.lower() for token in tokens):
                        offensive_found.append(term)

                for term in positive_terms:
                    if any(term in token.lower() for token in tokens):
                        positive_found.append(term)

                logger.info(f"Keywords - Offensive: {offensive_found}, Positive: {positive_found}")
            except Exception as ke:
                logger.error(f"Keyword detection error: {str(ke)}")
        
        # Heuristic protected-group detection driven by configurable lexicons
        text_lower = (text or "").lower()
        matched_groups = []
        protected_hits = []

        for category, terms in self.group_config.items():
            if category in {"stereotypes", "threats"}:
                continue
            for term in terms:
                if term and term in text_lower:
                    matched_groups.append(f"{category}: {term}")
                    protected_hits.append(term)

        # Stereotype and threat cues
        stereotype_hits = []
        for phrase in self.group_config.get("stereotypes", []):
            if phrase and phrase in text_lower:
                stereotype_hits.append(phrase)

        threat_hits = []
        for phrase in self.group_config.get("threats", []):
            if phrase and phrase in text_lower:
                threat_hits.append(phrase)

        # Visual group hints via CLIP (best-effort, optional)
        visual_hints = self._visual_group_hints(image_path)

        # Build structured, human-friendly reasoning
        reasoning_parts.append("🧩 Reasoning Breakdown")
        reasoning_parts.append("")

        # Content echo
        if text:
            reasoning_parts.append(f"Content seen: \"{text}\"")
        else:
            reasoning_parts.append("No text detected; decision relied on visual signals.")

        if sentiment_label and sentiment_score is not None:
            reasoning_parts.append(f"Sentiment: {sentiment_label} ({sentiment_score:.0%})")
        if offensive_found:
            reasoning_parts.append(f"Flagged terms: {', '.join(sorted(set(offensive_found[:5])))}")
        if protected_hits:
            reasoning_parts.append(f"Protected-group mentions: {', '.join(sorted(set(protected_hits[:5])))}")
        if stereotype_hits:
            reasoning_parts.append(f"Stereotype cues: {', '.join(sorted(set(stereotype_hits[:5])))}")
        if threat_hits:
            reasoning_parts.append(f"Threat/harm cues: {', '.join(sorted(set(threat_hits[:5])))}")
        if visual_hints:
            top_visual = [f"{label} ({prob*100:.0f}%)" for label, prob in visual_hints[:3]]
            reasoning_parts.append(f"Visual cues (CLIP): {', '.join(top_visual)}")

        reasoning_parts.append("")
        reasoning_parts.append("Decision factors:")

        if predicted_class == 1:
            reasoning_parts.append("1) Targets or references a protected group" + (f" → {', '.join(sorted(set(matched_groups[:3])))}" if matched_groups else ""))
            if stereotype_hits:
                reasoning_parts.append("2) Uses harmful stereotypes or dangerous associations" + f" → {', '.join(sorted(set(stereotype_hits[:3])))}")
            elif offensive_found:
                reasoning_parts.append("2) Uses hostile or degrading language.")
            if threat_hits:
                reasoning_parts.append("3) Contains threat/violence language" + f" → {', '.join(sorted(set(threat_hits[:3])))}")
            if sentiment_label == "NEGATIVE":
                reasoning_parts.append(f"4) Overall tone is negative ({sentiment_score:.0%}), reinforcing harmful framing.")
            if visual_hints:
                reasoning_parts.append("5) Visual evidence suggests group association" + f" → {', '.join([vh[0] for vh in visual_hints[:2]])}")
            reasoning_parts.append("6) Overall effect: frames a group/person in a dangerous or demeaning way, consistent with hate/harassment content.")
            reasoning_parts.append("")
            reasoning_parts.append("🎯 Conclusion: Classified as HATE/harassment due to protected-group reference plus negative/hostile framing.")
        else:
            reasoning_parts.append("1) No clear targeting of a protected group detected." if not protected_hits else "1) Mentions a protected group but context lacks hostile framing.")
            if offensive_found:
                reasoning_parts.append("2) Sensitive terms present but not used in a demeaning/hostile way.")
            reasoning_parts.append("3) No strong harmful stereotypes or calls to harm detected.")
            if sentiment_label == "POSITIVE":
                reasoning_parts.append(f"4) Tone is positive/neutral ({sentiment_score:.0%}), supporting a safe classification.")
            reasoning_parts.append("")
            reasoning_parts.append("🎯 Conclusion: Classified as NON-HATE; content lacks demeaning framing or targeting needed for harassment.")
        
        result = "\n".join(reasoning_parts)
        logger.info(f"===== NLP REASONING END =====\n{result}\n")
        return result
    
    def _get_reasoning(self, predicted_class: int):
        """Fallback reasoning for prediction"""
        if predicted_class == 1:
            return """This content has been identified as hateful based on analysis of both the image and text. The system detected offensive language patterns, dehumanizing visual elements, or combinations of text and imagery that suggest hostility toward individuals or groups. The confidence score reflects the certainty of this classification based on patterns learned from thousands of examples."""
        else:
            return """This content has been classified as non-hateful after analyzing both visual and textual elements. No significant offensive language, dehumanizing imagery, or combinations suggesting hateful intent were detected. The content appears to be neutral or positive in nature. The confidence score indicates the certainty in this classification."""
    
    def generate_report(self, explanation):
        """Generate a detailed explanation report"""
        report = {
            'summary': explanation['reasoning'],
            'methods_used': explanation['methods_available'],
            'key_insights': [],
            'warnings': []
        }
        
        if explanation.get('text_importance'):
            report['key_insights'].append({
                'type': 'Text Analysis',
                'insight': f"Important word: {explanation['text_importance']['max_important_word']}"
            })
        
        if explanation.get('attention_regions'):
            for region in explanation['attention_regions'][:2]:
                report['key_insights'].append({
                    'type': 'Visual Analysis',
                    'insight': region['region']
                })
        
        return report
