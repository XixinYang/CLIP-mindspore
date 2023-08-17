# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)


## Usage

First, [install MindSpore 2.0.0](https://www.mindspore.cn/install) (or later), as well as small additional dependencies, and then install this repo as a Python package.

```bash
$ pip install ftfy regex tqdm mindspore
$ pip install git+https://github.com/openai/CLIP.git
```

Example code of CLIP:

```python
import clip
from PIL import Image
from mindspore import Tensor,nn

model, preprocess = clip.load("./ViT-B-32.ckpt", device="Ascend")

image = Tensor(preprocess(Image.open("CLIP.png")))
text = clip.tokenize(["a diagram", "a dog", "a cat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = nn.Softmax(axis=-1)(logits_per_image).numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]
```


## API

The CLIP module `clip` provides the following methods:

#### `clip.available_models()`

Returns the names of the available CLIP models.

#### `clip.load(path, device=..., jit=False)`

Returns the model and the transform operations needed by the model, specified by the model name returned by `clip.available_models()`. The `path` argument is a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the Ascend device if there is any, otherwise the CPU.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a Tensor containing tokenized sequences of given text input(s). This can be used as the input to the model.

---

The model returned by `clip.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

#### `model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `model(image: Tensor, text: Tensor)`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.



## More Examples

### Zero-Shot Prediction

The code below performs zero-shot prediction using CLIP, as shown in Appendix B in the paper. This example takes an image from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), and predicts the most likely labels among the 100 textual labels from the dataset.

```python
import clip
from mindspore import ops, nn, Tensor
import mindspore.dataset as ds
from download import download
from PIL import Image

# Load the model
model, preprocess = clip.load("./ViT-B-32.ckpt", device="Ascend")

# Download the dataset
cifar100_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-100-binary.tar.gz"
download(cifar100_url, "./", kind="tar.gz", replace=True)
cifar100_iter = ds.Cifar100Dataset("cifar-100-binary", usage="test", shuffle=False)
cifar100=[]
for i in cifar100_iter:
    cifar100.append([Image.fromarray(i[0].asnumpy()),int(i[2])])

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = Tensor(preprocess(image))
text_inputs = ops.cat([clip.tokenize(f"a photo of a {i[1]}") for i in cifar100])

# Calculate features
image_features = model.encode_image(image_input)
text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = nn.Softmax(axis=-1)(100.0 * image_features @ text_features.T)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
index2label=[]
with open('./cifar-100-binar/fine_label_names.txt','r') as f:
	for line in f:
		index2label.append(line.strip('\n'))
for value, index in zip(values, indices):
    print(f"{index2label[index]:>16s}: {100 * float(value):.2f}%")
```

The output will look like the following (the exact numbers may be slightly different depending on the compute device):

```
Top predictions:

           snake: 65.31%
          turtle: 12.29%
    sweet_pepper: 3.83%
          lizard: 1.88%
       crocodile: 1.75%
```

Note that this example uses the `encode_image()` and `encode_text()` methods that return the encoded features of given inputs.


### Linear-probe evaluation

The example below uses [scikit-learn](https://scikit-learn.org/) to perform logistic regression on image features.

```python
import clip
from mindspore import ops
import mindspore.dataset as ds
from download import download
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Load the model
model, preprocess = clip.load("./ViT-B-32.ckpt", device="Ascend")

# Load the dataset
cifar100_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-100-binary.tar.gz"
download(cifar100_url, "./", kind="tar.gz", replace=True)
cifar100_test_iter = ds.Cifar100Dataset("cifar-100-binary", usage="test", shuffle=False)
cifar100_train_iter = ds.Cifar100Dataset("cifar-100-binary", usage="train", shuffle=False)
cifar100_test_iter = cifar100_test_iter.map(preprocess, input_columns=["image"])
cifar100_test_iter=cifar100_test_iter.batch(100)
cifar100_train_iter = cifar100_train_iter.map(preprocess, input_columns=["image"])
cifar100_train_iter=cifar100_train_iter.batch(100)

def get_features(dataset):
    all_features = []
    all_labels = []
    for images, _, labels in tqdm(dataset):
        features = model.encode_image(images)
        all_features.append(features)
        all_labels.append(labels)

    return ops.cat(all_features).asnumpy(), ops.cat(all_labels).asnumpy()


# Calculate the image features
train_features, train_labels = get_features(cifar100_train_iter)
test_features, test_labels = get_features(cifar100_test_iter)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
```

Note that the `C` value should be determined via a hyperparameter sweep using a validation split.


## See Also

* [OpenCLIP](https://github.com/mlfoundations/open_clip): includes larger and independently trained CLIP models up to ViT-G/14
