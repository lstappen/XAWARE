import torch
from torch import nn
from network.stand_alone_blocks import Bottleneck
from network.attention_blocks import Stack, FABlock
from network.inceptionresnet import inceptionresnetv2

def create_model(feature_types, inputs, head, pretrained):
    input_size = []
    if 'faces_img_extractor' in feature_types:
        input_size.append((3, 299, 299))
        face_i = inceptionresnetv2(pretrained=pretrained)
    else:
        face_i = None
    if 'env_img_extractor' in feature_types:
        input_size.append((3, 299, 299))
        env_i = inceptionresnetv2(pretrained=pretrained)
    else:
        env_i = None

    if inputs == 'all':
        input_size = [(3, 299, 299), (3, 299, 299), (164,), (350,)]
        if head == 'norm':
            print("Overall concat with single streams - norm ")
            model_ft = End2EndConCatNetNorm(modelA=face_i, modelB=env_i, inreshead=head)
        elif head[:2] == 'FA' or 'pool' in head:
            if '_daux' in head:
                print("Double Auxillery FancyAtt dual streams")
                model_ft = End2EndPCNetFullDoubleAux(modelA=face_i, modelB=env_i, head=head)
            elif '_aux' in head:
                print("Overall Auxillery FancyAtt dual streams")
                model_ft = End2EndPCNetFullAux(modelA=face_i, modelB=env_i, head=head)
            else:
                print("Overall FancyAtt dual streams")
                model_ft = End2EndPCNetFull(modelA=face_i, modelB=env_i, head=head)
        else:
            print("Overall concat with single streams ")
            model_ft = End2EndConCatNet(modelA=face_i, modelB=env_i, inreshead=head)

    if inputs == 'all_i':
        input_size = [(3, 299, 299), (3, 299, 299)]
        print(head)
        if 'v1' in head:
            print("Double Steam image and concat")
            model_ft = End2EndPCNetv1(modelA=face_i, modelB=env_i, head=head)
        elif head == 'inter':
            print("Double Steam image and Interaction fusion")
            model_ft = End2EndInterImgNet(modelA=face_i, modelB=env_i, head=head)
        else:
            print("Double Steam image and attention fusion")
            model_ft = End2EndPCNet(modelA=face_i, modelB=env_i, head=head)
    
    if 'all' not in inputs:
        if 'faces_img_extractor' in feature_types:
            if head is not None:
                print("Single image and attention fusion")
                model_ft = End2EndSingleImgStackAtt(modelA=face_i, head=head)
            else:
                print("Single Img Network for face. ")
                model_ft = End2EndImgNet(modelA=face_i)
        else:
            if head is not None:
                print("Single image and attention fusion")
                model_ft = End2EndSingleImgStackAtt(modelA=env_i, head=head)
            else:
                print("Single Img Network for env. ")
                model_ft = End2EndImgNet(modelA=env_i)

    return model_ft, input_size


class InResHead(nn.Module):
    def __init__(self, downsize=1024):
        super(InResHead, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=1024, kernel_size=(3, 3), stride=1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(36864, downsize),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(downsize, 256),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.head(x)
        return y


class GoCarHead(nn.Module):
    def __init__(self, dropout=0.5, downsize=128, activation='sigmoid'):
        super(GoCarHead, self).__init__()

        self.compressgocar = nn.Sequential(
            nn.Linear(350, downsize),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
            nn.Linear(downsize, 64))

        if activation == 'prelu':
            # some trys to avoid normalisation
            self.activation = nn.Sequential(
                nn.PReLU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.activation = nn.Sequential(
                nn.Sigmoid(),
                nn.Dropout(p=dropout),
            )

    def forward(self, x):
        y = self.compressgocar(x)
        y = self.activation(y)
        return y


class FaceFeaturesHead(nn.Module):
    def __init__(self, dropout=0.5, activation='sigmoid'):
        super(FaceFeaturesHead, self).__init__()

        self.compressff = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(164, 64),  # 740
        )
        if activation == 'prelu':
            # some trys to avoid normalisation
            self.activation = nn.Sequential(
                nn.PReLU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.activation = nn.Sequential(
                nn.Sigmoid(),
                nn.Dropout(p=dropout),
            )

    def forward(self, x):
        y = self.compressff(x)
        y = self.activation(y)
        return y


class InteractionBlock(nn.Module):
    def __init__(self, n_classes, input_size=740, dropout=0.1):
        super(InteractionBlock, self).__init__()

        self.interaction = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, 512),  # 740
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        y = self.interaction(x)
        return y


class ImageSepInResHead(nn.Module):
    def __init__(self, modelA=None, modelB=None, head=''):
        super(ImageSepInResHead, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        if head == 't_complex':
            print('head:  inreshead')
            self.head1 = InResHead()
            self.head2 = InResHead()
        elif head == 'standalone':
            self.head1 = Bottleneck(in_channels=1536, out_channels=256, stride=1)
            self.head2 = Bottleneck(in_channels=1536, out_channels=256, stride=1)

    def forward(self, env_img, face_img):
        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.head1(env_img)  # sigmoid
        face_img = self.head2(face_img)  # sigmoid
        return env_img, face_img


class ImageShareInResHead(nn.Module):
    def __init__(self, modelA=None, modelB=None, head=''):
        super(ImageShareInResHead, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        if head == 't_complex':
            print('head: inreshead')
            self.head = InResHead()
        elif head == 'standalone':
            self.head = Bottleneck(in_channels=1536, out_channels=256, stride=1)

    def forward(self, env_img, face_img):
        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.head(env_img)  # sigmoid
        face_img = self.head(face_img)  # sigmoid
        return env_img, face_img


class FeatureBlock(nn.Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()
        self.facefeatureshead = FaceFeaturesHead()
        self.gocarhead = GoCarHead()

    def forward(self, gocar, faces_extractor):
        gocar = self.gocarhead(gocar)
        faces_extractor = self.facefeatureshead(faces_extractor)

        return gocar, faces_extractor


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class End2EndSingleImgStackAtt(nn.Module):
    def __init__(self, modelA=None, head='', n_classes=9):
        super(End2EndSingleImgStackAtt, self).__init__()
        self.modelA = modelA
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3
        self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)

        if head == 'direct':
            post_precess = [
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes),
            ]

        elif head == 'FA_direct':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes)
            ]

        elif head == 'pool':
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, n_classes)]

        else:
            print("head not defined.", head)
            exit(1)

        self.post = nn.Sequential(*post_precess)
        self.pred = nn.Sequential(
            nn.PReLU(),
        )

    def forward(self, img):
        img = self.modelA(img)
        out = self.stack1(img)
        x = self.post(out)
        y = self.pred(x)
        return y


### SIMPLE INTERACTION MODELS ###
class End2EndInterImgNet(nn.Module):
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndInterImgNet, self).__init__()
        self.ImageStream = ImageSepInResHead(modelA, modelB, head)
        self.interaction = InteractionBlock(n_classes, input_size=512)

    def forward(self, env_img, face_img):
        env_img, face_img = ImageStream(env_img, face_img)
        x = torch.cat((env_img, face_img), 1)
        y = self.interaction(x)
        return y


class End2EndConCatNet(nn.Module):
    def __init__(self, modelA=None, modelB=None, inreshead='', n_classes=9):
        super(End2EndConCatNet, self).__init__()
        self.ImageStream = ImageSepInResHead(modelA, modelB, inreshead)
        self.FeatureStream = FeatureBlock()
        self.interaction = InteractionBlock(n_classes, input_size=640)

    def forward(self, env_img, face_img, faces_extractor, gocar):
        gocar, faces_extractor = self.FeatureStream(gocar, faces_extractor)
        env_img, face_img = self.ImageStream(env_img, face_img)
        x = torch.cat((env_img, face_img, faces_extractor, gocar), 1)
        y = self.interaction(x)
        return y


### ONE INPUT MODELS ###
class End2EndImgNet(nn.Module):
    def __init__(self, modelA=None, downsize=1024, n_classes=9):
        super(End2EndImgNet, self).__init__()
        self.modelA = modelA
        self.net = nn.Sequential(
            InResHead(downsize),
            nn.Linear(256, n_classes),
            nn.Softmax(),
        )

    def forward(self, img_ext):
        img_ext = self.modelA(img_ext)
        y = self.net(img_ext)
        return y


### TWO IMAGE INPUT MODELS ###
class End2EndPCNet(nn.Module):
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndPCNet, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3
        self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
        # self.stack2 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
        self.ga = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid(),

        ])

        if head == 'direct':
            post_precess = [
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes),
            ]

        elif head == 'FA_direct':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes)
            ]

        elif head == 'FA_direct_drop':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, n_classes)
            ]

        elif head == 'trad':
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                conv(512, 10, self.kernel_size),
                nn.Flatten(),
                nn.Linear(640, n_classes)]

        elif head == 'pool':
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, n_classes)]

        self.post = nn.Sequential(*post_precess)
        self.pred = nn.Sequential(
            nn.PReLU(),
        )

    def forward(self, env_img, face_img):
        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.stack1(env_img)

        face_img = self.stack1(face_img)
        # face_img = self.stack2(face_img)

        w = self.ga(torch.cat([env_img, face_img], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  #:, :
        out = w[:, 0, ::] * env_img + w[:, 1, ::] * face_img
        x = self.post(out)
        y = self.pred(x)
        return y


class End2EndPCNetFull(nn.Module):
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndPCNetFull, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3
        self.facefeatureshead = FaceFeaturesHead()
        self.gocarhead = GoCarHead()
        self.stack = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)

        self.interaction = InteractionBlock(n_classes, input_size=640)

        self.ga = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid(),

        ])  # nn.Linear(256, n_classes) ,

        if head == 'FA_direct':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.Sigmoid(),
            ]

        elif head == 'FA_direct_drop':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(98304, 512),
                nn.Sigmoid(),
            ]

        elif head == 'pool':
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Sigmoid()]

        self.post = nn.Sequential(*post_precess)
        self.pred = nn.Sequential(
            nn.PReLU(),
        )

    def forward(self, env_img, face_img, faces_extractor, gocar):  #
        gocar = self.gocarhead(gocar)
        faces_extractor = self.facefeatureshead(faces_extractor)

        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.stack(env_img)
        face_img = self.stack(face_img)

        w = self.ga(torch.cat([env_img, face_img], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  #:, :
        out = w[:, 0, ::] * env_img + w[:, 1, ::] * face_img

        x = self.post(out)
        x = torch.cat((x, faces_extractor, gocar), 1)
        x = self.interaction(x)

        # print(x.shape)
        y = self.pred(x)

        return y


class End2EndPCNetFullAux(nn.Module):
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndPCNetFullAux, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3
        self.facefeatureshead = FaceFeaturesHead()
        self.gocarhead = GoCarHead()
        self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
        self.stack2 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)

        self.interaction = InteractionBlock(n_classes, input_size=640)

        self.ga = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid(),

        ])  # nn.Linear(256, n_classes) ,

        if head == 'FA_direct' or '_aux' in head:
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.Sigmoid(),
            ]
            self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
            self.stack2 = self.stack1
        if head == 'FA_direct' or '_sepaux' in head:
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.Sigmoid(),
            ]
            self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
            self.stack2 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)

        elif head == 'FA_direct_drop':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(98304, 512),
                nn.Sigmoid(),
            ]
            self.stack1 = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
            self.stack2 = self.stack1

        self.post = nn.Sequential(*post_precess)
        self.aux_pred = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.PReLU(),
        )
        self.pred = nn.Sequential(
            nn.PReLU(),
        )

    def forward(self, env_img, face_img, faces_extractor, gocar):  #
        gocar = self.gocarhead(gocar)
        faces_extractor = self.facefeatureshead(faces_extractor)

        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.stack1(env_img)
        face_img = self.stack2(face_img)

        w = self.ga(torch.cat([env_img, face_img], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  #:, :
        out = w[:, 0, ::] * env_img + w[:, 1, ::] * face_img

        x = self.post(out)
        aux = self.aux_pred(x)

        z = torch.cat((x, faces_extractor, gocar), 1)
        z = self.interaction(z)

        z = self.pred(z)

        return z, aux


class End2EndPCNetFullDoubleAux(nn.Module):
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndPCNetFullDoubleAux, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3

        self.facefeatureshead = FaceFeaturesHead(activation='prelu')
        self.gocarhead = GoCarHead(activation='prelu')
        self.fuse = nn.Sequential(
            nn.Linear(96, 96),  # 740
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 64),
            nn.PReLU(),
        )
        self.aux_pred1 = nn.Sequential(
            nn.Linear(64, n_classes),
            nn.PReLU(),
        )

        self.stack = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)

        self.ga = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.PReLU(),
        ])  # nn.Linear(256, n_classes) ,

        if 'FA_direct' in head:
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
            ]

        elif 'pool' in head:
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.PReLU()]

        self.post = nn.Sequential(*post_precess)
        self.aux_pred2 = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.PReLU(),
        )  #

        self.pred = nn.Sequential(
            nn.Linear(576, n_classes),
            nn.PReLU(),
        )

    def forward(self, env_img, face_img, faces_extractor, gocar):  #
        gocar = self.gocarhead(gocar)
        faces_extractor = self.facefeatureshead(faces_extractor)
        z = torch.cat((faces_extractor, gocar), 1)
        z1 = self.fuse(z)
        aux1 = self.aux_pred1(z1)

        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)
        env_img = self.stack(env_img)
        face_img = self.block(face_img)

        w = self.ga(torch.cat([env_img, face_img], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  #:, :
        out = w[:, 0, ::] * env_img + w[:, 1, ::] * face_img

        z2 = self.post(out)
        aux2 = self.aux_pred2(z2)

        z = torch.cat((z1, z2), 1)

        # print(x.shape)
        z = self.pred(z)

        return z, aux1, aux2


# shared / v1`
class End2EndPCNetv1(nn.Module):
    # using grid attention in cross-modal representation learning in one block
    def __init__(self, modelA=None, modelB=None, head='', n_classes=9):
        super(End2EndPCNetv1, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.conv = conv
        self.dim = 1536
        self.gps = 2
        self.kernel_size = 3
        self.stack = Stack(self.conv, dim=self.dim, kernel_size=self.kernel_size)
        self.ga = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid(),

        ])

        if head == 'direct':
            post_precess = [
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes),
            ]

        elif head == 'FA_direct':
            post_precess = [
                FABlock(self.dim),
                nn.Flatten(),
                nn.Linear(98304, 512),
                nn.PReLU(),
                nn.Linear(512, n_classes)
            ]

        elif 'pool' in head:
            post_precess = [
                FABlock(self.dim),
                conv(self.dim, 512, self.kernel_size),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, n_classes)]

        self.post = nn.Sequential(*post_precess)
        self.pred = nn.Sequential(
            nn.PReLU(),
        )

    def forward(self, env_img, face_img):
        env_img = self.modelA(env_img)
        face_img = self.modelB(face_img)

        env_img = self.stack(env_img)
        face_img = self.stack(face_img)

        w = self.ga(torch.cat([env_img, face_img], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]  #:, :
        out = w[:, 0, ::] * env_img + w[:, 1, ::] * face_img
        x = self.post(out)
        y = self.pred(x)
        return y