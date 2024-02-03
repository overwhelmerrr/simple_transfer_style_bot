import logging
import cv2
import numpy as np
import os


from aiogram import Bot, Dispatcher
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from PIL import Image


# transfer model block
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 256  # use small size if no GPU

loader = transforms.Compose(
    [transforms.Resize(imsize), transforms.ToTensor()]  # scale imported image
)  # transform it into a torch tensor


def image_loader(image):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Resize and transform the image
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise, the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input_):
        self.loss = F.mse_loss(input_, self.target)
        return input_


def gram_matrix(input_):
    a, b, c, d = input_.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input_.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input_):
        G = gram_matrix(input_)
        self.loss = F.mse_loss(G, self.target)
        return input_


cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


# create a module to normalize input image, so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=400,
    style_weight=1000000,
    content_weight=1,
    result_path="result.jpg",
):

    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    # print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)
        result_img = input_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        result_img = (result_img * 255).astype(np.uint8)
        cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    return result_path


API_TOKEN = os.environ["API_TOKEN"]
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

logging.basicConfig(level=logging.INFO)


# FSM for states
class YourState(StatesGroup):
    waiting_for_first_photo = State()
    waiting_for_second_photo = State()


# basic start handler
@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    chat_id = message.chat.id  # Get the chat ID
    subfolder_path = f"photos/{chat_id}"
    os.makedirs(subfolder_path, exist_ok=True)  # Create subfolder if it doesn't exist

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item = types.KeyboardButton("Transfer Style")
    markup.add(item)

    await message.answer("Hello! I'm transfer style bot. Please, press the button to continue:", reply_markup=markup)


# strict handler for all states
@dp.message_handler(commands=['transfer_style'], state='*')
async def transfer_style_main(message: types.Message):
    await message.answer("Send photo of 'content' (send as photo, documents are not supported now)")
    await YourState.waiting_for_first_photo.set()

# Register the command handler
dp.register_message_handler(transfer_style_main, commands=['transfer_style'])


# main handler
@dp.message_handler(lambda message: message.text == "Transfer Style")
async def transfer_style(message: types.Message):
    await message.answer("Send photo of 'content' (send as photo, documents are not supported now)")
    await YourState.waiting_for_first_photo.set()


# first photo handler
@dp.message_handler(content_types=types.ContentType.PHOTO, state=YourState.waiting_for_first_photo)
async def process_first_photo(message: types.Message, state: FSMContext):
    chat_id = message.chat.id
    subfolder_path = f"photos/{chat_id}"

    photo = message.photo[-1]
    photo_id = photo.file_id

    photo_path = f"{subfolder_path}/{photo_id}.jpg"
    await photo.download(photo_path)

    await YourState.next()
    await state.update_data(first_photo_path=photo_path)

    await message.answer("Content has been received. Please send 'style' photo.")


# second photo handler
@dp.message_handler(content_types=types.ContentType.PHOTO, state=YourState.waiting_for_second_photo)
async def process_second_photo(message: types.Message, state: FSMContext):
    chat_id = message.chat.id
    subfolder_path = f"photos/{chat_id}"

    photo = message.photo[-1]
    photo_id = photo.file_id

    photo_path = f"{subfolder_path}/{photo_id}.jpg"
    await photo.download(photo_path)

    await message.answer("Style photo has been received. Processing will take up to 5 minutes...")

    data = await state.get_data()
    first_photo_path = data.get('first_photo_path')

    img1 = cv2.imread(first_photo_path)
    img2 = cv2.imread(photo_path)

    # resize is mismatch size
    if img1.shape != img2.shape:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])

        img1 = cv2.resize(img1, (min_width, min_height))
        img2 = cv2.resize(img2, (min_width, min_height))

    img1 = image_loader(img1)
    img2 = image_loader(img2)

    img3 = img1.clone()

    # transfer style
    result_image = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, img1, img2, img3, result_path=f"{subfolder_path}/result.jpg")

    await message.answer("Processing finished. Here is the result:")
    with open(result_image, 'rb') as photo:
        await bot.send_photo(message.from_user.id, photo)

    await state.finish()
    await message.answer("If you want to continue, press the button and send new images.")


if __name__ == '__main__':
    from aiogram import executor
    from aiogram.contrib.fsm_storage.memory import MemoryStorage

    storage = MemoryStorage()
    dp.storage = storage

    executor.start_polling(dp, skip_updates=True)
