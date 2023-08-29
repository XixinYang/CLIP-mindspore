import clip
from PIL import Image
from mindspore import Tensor,nn, ops
import mindspore as ms
import mindspore.dataset as ds


ms.set_context(device_target="CPU", mode=1)
model, preprocess = clip.load("RN50", device="CPU")
#
# cifar100_iter = ds.Cifar100Dataset("C:\\Users\\Yang Xixin\\PycharmProjects\\CLIP-mindspore\\cifar-100-binary", usage="test", shuffle=False)
# cifar100=[]
# for i in cifar100_iter:
#     cifar100.append([Image.fromarray(i[0].asnumpy()),int(i[2])])
#
# # Prepare the inputs
# image, class_id = cifar100[3637]
# image_input = Tensor(preprocess(image))
# text_inputs = ops.cat([clip.tokenize(f"a photo of a {i[1]}") for i in cifar100])
#
# # Calculate features
# image_features = model.encode_image(image_input)
# text_features = model.encode_text(text_inputs)
#
# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = nn.Softmax(axis=-1)(100.0 * image_features @ text_features.T)
# values, indices = similarity[0].topk(5)
#
# # Print the result
# print("\nTop predictions:\n")
# index2label=[]
# with open('./cifar-100-binar/fine_label_names.txt','r') as f:
# 	for line in f:
# 		index2label.append(line.strip('\n'))
# for value, index in zip(values, indices):
#     print(f"{index2label[index]:>16s}: {100 * float(value):.2f}%")
# print("a")


image = Tensor(preprocess(Image.open("CLIP.png")))
text = clip.tokenize(["a diagram", "a dog", "a cat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = nn.Softmax(axis=-1)(logits_per_image).numpy()

print("Label probs:", probs)




