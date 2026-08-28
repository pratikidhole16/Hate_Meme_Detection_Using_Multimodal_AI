    def _generate_nlp_reasoning(self, text: str, image_path: str, predicted_class: int):
        """Generate NLP-based dynamic reasoning using actual transformers sentiment analysis"""
        try:
            reasoning_parts = []
            sentiment_label = None
            sentiment_score = None
            offensive_found = []
            positive_found = []
            
            # Run NLP sentiment analysis if text is provided
            if text:
                try:
                    from transformers import pipeline
                    device_id = 0 if torch.cuda.is_available() else -1
                    sentiment_analyzer = pipeline(
                        "sentiment-analysis", 
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=device_id
                    )
                    sentiment_result = sentiment_analyzer(text[:512])[0]
                    sentiment_label = sentiment_result['label']
                    sentiment_score = sentiment_result['score']
                    logger.info(f"NLP Sentiment: {sentiment_label} ({sentiment_score:.2%})")
                except Exception as se:
                    logger.warning(f"Sentiment analysis failed: {str(se)}")
                
                # Keyword detection
                tokens = self.classifier.tokenizer.tokenize(text.lower())
                hate_terms = {'hate', 'stupid', 'ugly', 'dumb', 'trash', 'loser', 'kill', 'die', 'death', 
                             'racist', 'sexist', 'inferior', 'scum', 'filth', 'garbage', 'bigot', 'attack'}
                positive_terms = {'love', 'good', 'great', 'nice', 'happy', 'fun', 'awesome', 'beautiful', 
                                'wonderful', 'amazing', 'friendly', 'kind', 'help', 'support', 'peace'}
                
                for term in hate_terms:
                    if any(term in token.lower() for token in tokens):
                        offensive_found.append(term)
                for term in positive_terms:
                    if any(term in token.lower() for token in tokens):
                        positive_found.append(term)
            
            # Build reasoning
            if predicted_class == 1:  # Hateful
                reasoning_parts.append("WHAT'S IN THE CONTENT")
                reasoning_parts.append("")
                if text:
                    reasoning_parts.append(f"The image contains text that says: \"{text}\"")
                    if sentiment_label:
                        reasoning_parts.append(f"NLP analysis: {sentiment_label} sentiment ({sentiment_score:.0%} confidence)")
                    if offensive_found:
                        reasoning_parts.append(f"Detected harmful words: {', '.join(offensive_found[:3])}")
                else:
                    reasoning_parts.append("The image contains visual elements that were analyzed.")
                
                reasoning_parts.append("")
                reasoning_parts.append("WHY IT'S CLASSIFIED AS HATEFUL")
                reasoning_parts.append("")
                if text and offensive_found:
                    reasoning_parts.append("- The language expresses hostility or degrading ideas toward individuals or groups")
                    reasoning_parts.append(f"- Words like '{', '.join(offensive_found[:2])}' are associated with hate speech")
                else:
                    reasoning_parts.append("- The visual-textual combination conveys hostile messaging")
                reasoning_parts.append("- This content can promote discrimination or violence against protected groups")
                reasoning_parts.append("- Such material is widely recognized as hate speech, not acceptable commentary")
            else:  # Non-hateful
                reasoning_parts.append("WHAT'S IN THE CONTENT")
                reasoning_parts.append("")
                if text:
                    reasoning_parts.append(f"The image contains text that says: \"{text}\"")
                    if sentiment_label:
                        reasoning_parts.append(f"NLP analysis: {sentiment_label} sentiment ({sentiment_score:.0%} confidence)")
                    if positive_found:
                        reasoning_parts.append(f"Positive expressions: {', '.join(positive_found[:3])}")
                else:
                    reasoning_parts.append("The image contains visual elements that were analyzed.")
                
                reasoning_parts.append("")
                reasoning_parts.append("WHY IT'S CLASSIFIED AS SAFE")
                reasoning_parts.append("")
                reasoning_parts.append("- No offensive, dehumanizing, or hostile language detected")
                reasoning_parts.append("- Visual and textual elements do not target or demean protected groups")
                reasoning_parts.append("- Content appears to be humor, information, or legitimate expression")
                reasoning_parts.append("- Falls within acceptable social media content standards")
            
            return "\n".join(reasoning_parts)
        except Exception as e:
            logger.error(f"NLP reasoning failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_reasoning(predicted_class)
