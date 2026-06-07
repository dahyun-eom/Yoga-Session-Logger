# YoseLog 10-Minute Presentation Script

## Slide 1 — Title
Hi, my project is YoseLog, a real-time yoga session logger. The goal is not only to recognize poses from a webcam, but to turn those predictions into a usable workout/session record. The final system classifies yoga poses, confirms stable holds, estimates balance, logs duration automatically, and shows everything in a web dashboard.

## Slide 2 — Problem
Many yoga applications are built around a pre-selected pose. The user chooses the pose first, then the app measures how well the user matches that pose and gives scoring or feedback. That can work for pose correction, but it is not ideal for a free yoga session, because the user has to keep telling the app what pose they are about to do. My idea was: what if the user can just do the session naturally, and the application automatically detects the pose, confirms it, and logs the whole session?

## Slide 3 — System Architecture
The system starts with OpenCV capturing webcam frames. MediaPipe Pose extracts 33 body landmarks, which become 66 x/y features. Those features go into the trained scikit-learn model. Then the session engine applies the same stability logic from classify_live.py, computes hold duration and balance, and sends the state to the Flask frontend. A key design decision was making the frontend display the confirmed result from the classifier loop, instead of inventing a separate frontend interpretation.

## Slide 4 — Why I Chose an MLP
I chose an MLP because my model input is not raw images. MediaPipe already converts each image into structured body landmarks, so the classifier receives 66 numeric features: x and y for 33 points. For this type of tabular landmark data, an MLP is a practical choice because it can learn nonlinear relationships between joints, such as how shoulders, hips, knees, and ankles are positioned relative to each other. It is also much lighter than training a CNN on raw images, and it is easier to iterate on for this project. Compared with hard-coded angle rules, the MLP can learn from examples and is easier to expand to 40 poses.

## Slide 5 — Classifier Pipeline
For the classifier, I downloaded images for 40 yoga poses and used MediaPipe Pose to extract 33 body landmarks from each image. Each sample becomes 66 numeric features, x and y for each landmark, and those rows are saved in keypoints.csv. A key improvement was that during keypoint extraction, I also created a horizontally flipped version of each image using cv2.flip and extracted landmarks from that flipped image too. This gave the model more left/right orientation coverage and helped bring the accuracy to around 87 percent. I wrote train_model_mlp.py to train an MLPClassifier with hidden layers of 256 and 128. The final dataset had 11,566 samples, split into 9,252 training samples and 2,314 testing samples. The model reached 86.9 percent accuracy on the held-out test set, and the trained model is saved as pose_model.pkl. At runtime, webcam landmarks are converted into the same 66-feature format, smoothed over five frames, and passed into model.predict.

## Slide 6 — Model Tuning Experiment
I also tested different MLP hidden-layer architectures. This helped answer the question of whether 256 and 128 was arbitrary. The results showed that most reasonable models clustered around 86 to 87 percent accuracy. The single hidden layer with 128 neurons was slightly highest at about 87.0 percent, while 256 and 128 reached about 86.9 percent, almost tied. I kept 256 and 128 because the accuracy difference was tiny, and 256 and 128 had slightly stronger macro-F1. That matters because my 40-pose dataset is imbalanced, so macro-F1 better reflects performance across rare classes as well as common classes. Larger models like 512 and 256 did not clearly improve accuracy, and the three-layer model performed worse. So my conclusion is that hidden-layer size matters, but more layers or more neurons are not automatically better.

## Slide 7 — Session Logging Logic
The session logger makes predictions usable. It compares the current smoothed keypoints to the previous smoothed keypoints. If the predicted pose is the same and the movement is under 0.05, the stable frame counter increases. The system requires 30 stable frames, which is about one second. Only then is the pose treated as a confirmed hold. When the pose changes, the previous pose is logged if it lasted at least one second.

## Slide 8 — Balance Scoring
Balance scoring is separate from classification. It does not judge whether the yoga form is correct. It estimates steadiness during a held pose. I store the stable keypoints during the hold, then measure torso center movement and shoulder tilt variation. Less sway means a higher balance score. This is important to explain clearly: balance score is a stability metric, not a yoga-correction metric.

## Slide 9 — Frontend Integration
The frontend turns the classifier into an application. It shows the live video stream, confirmed classifier result, hold timer, stability progress, balance score, and session log. The backend owns camera and inference; the browser only displays the state. During testing, I found that extra frontend rules made results differ from classify_live.py, so I aligned the web result back to the same classifier mechanism.

## Slide 10 — My Contribution
My contribution has two parts: the classifier work and the product work. For the classifier, I downloaded the dataset for 40 poses, extracted MediaPipe keypoints, added horizontal flip augmentation during extraction, wrote train_model_mlp.py, trained the MLP model, and evaluated it at 86.9 percent accuracy. For the product, I built the Flask web app, state API, live stream, dashboard, CSV export, and balance scoring. I also debugged the difference between raw model behavior and frontend behavior, and restored the frontend to follow the reliable classify_live.py path.

## Slide 11 — Limitations and Defense
There are still limitations. Similar poses can be confused, especially because this is a closed-set classifier. There is no explicit no-pose/resting class yet. Balance measures stillness, not correctness. Camera angle and body visibility matter. My defense is that I separated raw classifier testing from web logging, used stable-frame confirmation, kept the frontend aligned with the classifier, and identified clear next steps: add no-pose training data, collect more examples for confusing poses, and add pose-correctness scoring.

## Slide 12 — Final Takeaway
The final takeaway is that YoseLog turns pose detection into a session record. The user can see the confirmed pose, duration, balance score, and session log automatically. My main contribution is the reliable session layer around the classifier: taking a model output and building the application logic needed for a usable yoga logging experience.

# Professor Defense Q&A

## Q: Why does the classifier sometimes choose a pose when the user is resting?
Because the model is closed-set. It only knows the trained pose classes, so it must pick the nearest one. A better future version would include a no-pose/resting class.

## Q: Is the balance score measuring pose correctness?
No. It measures steadiness during a held pose using torso center sway and shoulder tilt variation. Correctness would require comparing joint angles to an ideal pose template.

## Q: Why did you align the frontend to classify_live.py?
Because the frontend should not become a second classifier. When extra UI gates changed results, I made the backend follow the same mechanism as the trusted live classifier loop.

## Q: What did you personally contribute?
I downloaded and prepared the 40-pose dataset, extracted MediaPipe keypoints, added flipped-version augmentation, wrote the MLP training script, trained/evaluated the classifier, and then built the web application layer: session state API, live dashboard, CSV export, balance scoring, raw classifier tester, and the debugging/alignment work that made the frontend reflect the classifier correctly.

## Q: What would you improve next?
I would add a no-pose/resting dataset class, collect more examples for visually similar poses, add per-pose correctness scoring using joint angles, and evaluate accuracy with live webcam test videos instead of only image-based validation.

## Q: Why use hidden layers of 256 and 128 if 128 was slightly higher in the experiment?
The difference was very small: around 87.0 percent versus 86.9 percent. I kept 256 and 128 because it had almost the same accuracy but slightly better macro-F1. Since the dataset is imbalanced, macro-F1 is important because it gives rare poses equal weight. I would explain it as a balanced choice, not the only possible choice. Future work could select the final architecture by cross-validation.

## Q: Why did you choose MLP instead of a CNN?
Because the input to my classifier is not raw images. MediaPipe already extracts structured landmark features, so the model receives 66 numeric values. An MLP is appropriate for this compact tabular feature vector, while a CNN would be more useful if I were training directly on raw pixels.
