# LuminSkin
The goal of this research is to determine the potential of Vision Transformers to accurately classify skin cancer using medical imaging. 
**Initially began as a college semester project, it evolved into a prospective research study.**
_______

# **Abstract**
Skin cancer is one of the most prevalent and fatal cancers, and diagnostic methods are therefore effective in making the intervention timely. Among such categories of skin cancer are melanoma, actinic keratosis, basal cell carcinoma, squamous cell carcinoma, and Merkel cell carcinoma, each being morphologically distinct to make them hard to detect and classify. Such cancers have a higher risk associated with them, as in the case of melanoma, which is an aggressive cancer and even unpredictive. It progresses quickly, and the chances of dying from it are high unless diagnosed in early stages. The chance for prompt treatment improves the patient's outcome and minimizes the chances of metastasis. Nevertheless, automatic classification of skin lesions faces enormous challenges owing to variabilities in lesions' appearance, color, texture, and shape that manifest differently in various forms and stages. It thus requires sophisticated computational tools able to capture all these complexities.

Traditionally, deep CNNs are applied to the classification of medical images and have proved to attain very high success rates for tasks like that of classification of skin lesions. The reason why such architecture is good at those applications is that it is particularly suited to identify local patterns in images, which helps make the specific features of the skin lesions arise and be captured. Very recently though, transformer-based architecture, specifically Vision Transformers, came into the scene that opened a different avenue for the classification tasks of medical images. Unlike CNNs, which are typically local feature extractors, self-attention in ViTs help model the long-range dependencies in an image. This ability potentially lets the ViTs learn complex pixel relationships and attain better performance at classifying visually heterogeneous skin lesions.

This paper presents the ability of Vision Transformers for classification of skin cancer from images. This research verifies if ViT models are trained on the curated ISIC Archive datasets of benign and malignant skin lesion varieties can classify with higher accuracy than state-of-the-art CNN-based approaches. The ISIC Archive comprises rich variability in images related to benign and malignant skin lesions. Using this approach, the study will establish whether this global-level feature representation capability in ViTs can lead to improved diagnostic accuracy in the detection of skin cancer. The outcome of this research will enhance automatic diagnostic support and enable health practitioners to make faster decisions with higher accuracy to assist in early detection and treatment of the disease for patients affected by skin cancer.

Index Terms — Vision Transformers, PyTorch, Skin Image Analysis, Skin Cancer, AI in medical
_______

![image](https://github.com/user-attachments/assets/97cdc857-f141-4cc2-95cc-d09efa7cf2bb)
<br>Fig.1 Vision Transformer Layman Illustration

_______

# **OBJECTIVES**
  •	Develop a particularly fine-tuned skin-cancer-detection ViT model, focusing on high accuracy in distinguishing between benign and malignant skin lesions.
  •	Leverages self-attention mechanisms in the ViTs for skin lesions to capture global dependencies and subtle patterns and thereby improve diagnostic precision beyond the traditional CNN-based methods.
  •	Train and test it using a clinically annotated skin cancer dataset, such as the ISIC Archive, so that it delivers clinically relevant accuracy across types of skin lesions, including melanoma.
  •	Use data augmentation techniques, since medical datasets are scarce and have limited sizes. This should be utilized to make the model more robust and reduce potential side effects of overfitting.
  •	Optimize the ViT model's computational efficiency such that it would scale to afford real-time applications in clinical environments.
  •	Ease interpretation of the model through availability of visualizations such as the attention map visualization of the areas in images that caused a prediction by the model, and thus ease clinician trust and transparency.
  •	Measure the performance of the model with metrics such as sensitivity, specificity, accuracy, and AUC for high reliability in diagnostics.

_______

# **Architecture Overview**

The ViT architecture provides a powerful alternative to CNNs in the task of image classification. Being based on CNNs where this architecture captures spatial relationships locally, the transformer architecture may be exploited to create global dependencies across an entire image, thus enabling ViTs to identify intricate relations among pixels over long distances-advantageous for medical imaging and others requiring fine-grained analysis.

1. Image Patching and Tokenization: In the ViT architecture, the first stage is to divide an input image into a sequence of smaller non-overlapping patches, in terms of pixels. For example, an input image of size 256 x 256 pixels can be divided into 16 x 16 patches, in other words, each of size 16 x 16 pixels, which, as a result, leads to a sequence of 256 patches. As soon as flattened, each patch gets turned into a 1D vector. This is done by transforming the patch into a sequence of linearly arranged pixel values. The patch vectors are then forwarded into a learnable linear projection layer that transforms each patch into an embedding dimension, for instance, into 1024 dimensions acting like tokens within the transformer model.

2. Class Token and Position Embeddings: One of the quite distinctive features of the ViT architecture is that there is a particular learnable embedding introduced as the class token. It added at the beginning of the sequence of patch embeddings, and it essentially plays the role of a global representation of the image and is eventually used for classification. Also, each patch embedding in ViT includes adding position embeddings to them. Since transformers are inherently position-agnostic, these position embeddings encode the spatial location of each patch within the image, which then helps the model understand the spatial structure of the input.

3. Transformer Encoder Blocks: After being tokenized, the patch embeddings, accompanied by the class token, are passed into the transformer encoder. The transformer architecture consists of two main parts in each of its encoder blocks:
There are two modules: MHSA and FFN. The components are repeated across multiple layers or transformer blocks with residual connections and layer normalization for stability during training and easier gradient flow.
<br>**•	Multi-Head Self-Attention (MHSA):** MHSA enables the model to learn the relationships among different patches by assigning attention weights, which allows it to focus on important patches across the image. The term "multi-head" indicates that a model contains multiple attention heads that can simultaneously look at different aspects of the image, capturing various interactions of features. For instance, a 16-head ViT model can process 16 different relationships in parallel, thereby taking a global view that is comprehensive.
<br>**•	Feed-Forward Neural Network (FFN):** This is another position-wise feed-forward network in each transformer block after the attention mechanism. Each token undergoes linear transformations along a series, which promotes nonlinear patterns by the model and obtains stronger representations of advanced features from the data.

5. Class Token Aggregation: From this processed sequence of patch embeddings, by feeding it to the transformer encoder, the transformer encoder output corresponding to the class token is extracted. Since the class token has acquired all the information coming from all the patches of the image due to the self-attention mechanism, it serves as the global representation of the whole image.

6. Classification Head: The class token is passed to the classification head. The classification head is a fully connected layer. Therefore, the classification head maps the representation of that class token to the output logits-the probability scores for every one of the classes involved in classification. If there is skin cancer classification involved, such as benign versus malignant, the classification head usually uses a SoftMax layer that provides normalized class probabilities.

7. Training and Optimization: The ViT model is trained with a suitable loss function for the considered classification task, such as cross entropy loss in binary or multiclass classification and uses optimization algorithms like Adam with techniques like learning rate scheduling and regularization to stabilize and optimize the training process.

![image](https://github.com/user-attachments/assets/a523582a-530c-416d-85ce-c48a84f11406)
<br>Fig.2 Vision Transformer Full Architecture

_______

# **DATASET DESCRIPTION**

The dataset used for this study was from the ISIC Archive, one of the most high-performing and established archives when it comes to dermatological research, particularly in skin lesion images. ISIC Archive provides a rich collection of curated datasets, supporting the development and validation of automated diagnostic tools in dermatology. This dataset is one of the most vital collections of data that can be used in assessing skin cancer due to an abundance of labeled images both benign and malignant lesions.

For this study, the dataset consists of a total of 3,600 images; they are split equally between the two main classes, one being benign and the other malignant skin lesions, thus having 1,800 images each of them. This guarantees equal representations for classes, a very important aspect in training supervised learning models to classify the skin lesions correctly. Having an equal number of images for each class reduces the risk that could lead to any class imbalance problems, biases the model toward more frequently occurring categories, and lowers its performance on less-represented classes.

All images in the dataset are already prelabeled and therefore fit the various supervised learning approaches, thus effectively training and testing the deep learning models. It further increases the credibility of the dataset, since for all lesions, either benign or malignant, it is tagged by experts. High-quality labeling ensures that more precise examples of what the model should learn between benign and malignant lesions are obtained.

The images are standardized to deep learning frameworks, so the CNNs and Vision Transformers will process them well. Consistency of image size in terms of resolution is important: it ensures that relevant features considered by the model are not induced by variations in size or quality. Compatibility with pre-trained models also needs to be assured; they are quite often used in fine-tuning and have specific requirements for input.

Every image in the ISIC Archive dataset is controlled qualitatively following standardized imaging protocols, for instance, lighting, focus, and skin tones diversity. The results have high quality images with diversity across different skin types and lesion characteristics, and this makes the model more robust for real world applications.

The ISIC Archive dataset of this study has a nicely labeled, balanced, and high-quality foundation in training and testing deep learning models aiming to detect skin cancer. It is standardized and spans a completely comprehensive range of benign and malignant cases, best suited for validation in the effectiveness of Vision Transformers and other deep learning architectures in medical image analysis.

![image](https://github.com/user-attachments/assets/2f3ad93b-5de7-4b8e-89e0-afd346f5444f)
<br>Fig.3 Distribution of Images before Data Augmentation, Test set [ LEFT ] and Train set [ RIGHT ]

_______

# **DATA AUGMENTATION**

Data augmentation, thus, holds importance for increased diversity of the training dataset and reducing overfitting with an improvement in the model's generalization to new data. It is obtained by generating the differently transformed versions of the original images, through which the model encounters all such transformations that it would witness in real life. In this experiment, a variety of data augmentation techniques are used with specific strengths to be sought in the datasets.

  •	Random Horizontal Flip: This looks to flip images horizontally with a probability of 0.5. It introduces variety in the dataset so that features can easily be identified by the model whether they are upward or downward-oriented, especially in cases where lesion patterns appear symmetrically.

  •	Random Rotation (±30°): The images are randomly rotated by ±30° so that the model can be invariant to orientation. Skin lesions will come in different orientations and at times may not be well aligned so that this rotation would ensure the feature extraction for patterns would occur despite the variations in orientation.

  •	Color Jittering: Applying random brightness, contrast, saturation, and hue alterations of the images to simulate photos taken at various lighting conditions. This will help make the model less sensitive to lighting differences and skin color differences that are typically dramatic in clinical settings. Thus, the model pays more attention to structural and textural characteristics than to color-related characteristics of the lesion.

  •	Randomly resized crops (Scale 0.8-1.0): This augmentation randomly crops images and scales between 0.8 and 1.0 to vary the features scale. This will make the model more generalizable to images of all sizes, reducing overfitting in lesion recognition at a close-up or distant view.

  •	The Random Affine and Perspective Transformations: Affine transformations (such as translation, scaling, rotation, etc.) and change of perspective simulate changes in scale and viewpoint. Such transformations make the model much more robust against slight positional and shape variations, thus further increasing its ability to generalize. The slight variance in imaging angles does not make it seem overly sensitive to certain orientations or placements within the frame.

These augmentations collectively enrich the training set in simulating a wide range of conditions by images. The model, in turn, captures more representative patterns and features in diverse scenarios, which helps reduce the risk of overfitting and improves performance on unseen data. By creating a varied dataset, these augmentations contribute toward building a more reliable and versatile diagnostic tool for skin cancer.

![image](https://github.com/user-attachments/assets/203d2e02-07c8-451a-9936-69bffd7e418a)
<br>Fig.4 Visualizing samples from “BENIGN” augmented dataset

![image](https://github.com/user-attachments/assets/f74c9142-4502-4a3e-9f0f-d9e3cc65fa17)
<br>Fig.5 Visualizing samples from “MALIGNANT” augmented dataset


