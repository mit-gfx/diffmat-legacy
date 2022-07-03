import torch as th
from torchvision.models.vgg import vgg19

class TextureDescriptor(th.nn.Module):

	def __init__(self, device):
		super(TextureDescriptor, self).__init__()
		self.device = device
		self.outputs = []

		# get VGG19 feature network in evaluation mode
		self.net = vgg19(True).features.to(device)
		self.net.eval()

		# change max pooling to average pooling
		for i, x in enumerate(self.net):
			if isinstance(x, th.nn.MaxPool2d):
				self.net[i] = th.nn.AvgPool2d(kernel_size=2)

		def hook(module, input, output):
			self.outputs.append(output)

		#for i in [6, 13, 26, 39]: # with BN
		for i in [4, 9, 18, 27]: # without BN
			self.net[i].register_forward_hook(hook)

		# weight proportional to num. of feature channels [Aittala 2016]
		self.weights = [1, 2, 4, 8, 8]

		# this appears to be standard for the ImageNet models in torchvision.models;
		# takes image input in [0,1] and transforms to roughly zero mean and unit stddev
		self.mean = th.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
		self.std = th.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

	def forward(self, x):
		self.outputs = []

		# run VGG features
		x = self.net(x)
		self.outputs.append(x)

		result = []
		batch = self.outputs[0].shape[0]

		for i in range(batch):
			temp_result = []
			for j, F in enumerate(self.outputs):
				F_slice = F[i,:,:,:]
				f, s1, s2 = F_slice.shape
				s = s1 * s2
				F_slice = F_slice.view((f, s))

				# Gram matrix
				G = th.mm(F_slice, F_slice.t()) / s
				temp_result.append(G.flatten())
			temp_result = th.cat(temp_result)

			result.append(temp_result)
		return th.stack(result)

	def eval_CHW_tensor(self, x):
		"only takes a pytorch tensor of size B * C * H * W"
		assert len(x.shape) == 4, "input Tensor cannot be reduced to a 3D tensor"
		x = (x - self.mean) / self.std
		return self.forward(x.to(self.device))