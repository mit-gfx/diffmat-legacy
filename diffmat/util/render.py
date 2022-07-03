import torch as th
import numpy as np

# utility functions 
def arr(*x):
	return th.tensor(x, dtype=th.float32)

def dstack(img, a):
	return th.stack([img * x for x in a], 2)

def roll0(A, n):
	return th.cat((A[n:, :], A[:n, :]), 0)

def roll1(A, n):
	return th.cat((A[:, n:], A[:, :n]), 1)

def normalize_all(x):
	s = th.sqrt((x ** 2).sum(2))
	return x / th.stack((s, s, s), 2)

def beckmann_ndf(cos_h, alpha):
	c2 = cos_h ** 2
	t2 = (1 - c2) / (c2 + 1e-8)
	a2 = alpha ** 2
	return th.exp(-t2 / a2) / (np.pi * a2 * c2**2 + 1e-8) # add 1e-8 to avoid zero-division

def ggx_ndf(cos_h, alpha):
	mask = cos_h > 0.0
	c2 = cos_h ** 2
	t2 = (1 - c2) / (c2 + 1e-8)
	a2 = alpha ** 2
	denom = np.pi * c2**2 * (a2 + t2)**2 + 1e-8  # add 1e-8 to avoid zero-division
	return a2 * mask / denom

def schlick(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def brdf(n_dot_h, alpha, f0):
	D = ggx_ndf(n_dot_h, alpha)
	# return f0 * D / (4.0 * n_dot_h**2 + 1e-8) # add 1e-8 to avoid zero-division
	return f0 * D / 4.0 # set n_dot_h**2 as geometry function

def render(normal, albedo, rough, metallic, light_color, f0, size, camera, diffuse_only_flag=False):
	# Remove alpha channel for normal and albedo
	if len(normal.shape) == 3 and normal.shape[0] == 4:
		normal = normal[:3,:,:]

	if len(albedo.shape) == 3 and albedo.shape[0] == 4:
		albedo = albedo[:3,:,:]

	# assume albedo in gamma space
	albedo = albedo ** 2.2

	axis = 0
	n = normal.shape[1]
	device = normal.device
	
	# update albedo using metallic
	f0 = f0 + metallic * (albedo - f0)
	albedo = albedo * (1.0 - metallic) 

	# n points between [-size/2, size/2]
	x = th.arange(n, dtype=th.float32, device=device)
	x = ((x + 0.5) / n - 0.5) * size

	# surface positions
	y, x = th.meshgrid((x, x))
	z = th.zeros_like(x)
	pos = th.stack((x, -y, z), axis)

	# directions (omega_in = omega_out = half)
	omega = camera - pos
	dist_sq = (omega ** 2).sum(axis)
	d = th.sqrt(dist_sq)
	omega = omega / (th.stack((d, d, d), axis) + 1e-8)

	# geometry term and brdf
	n_dot_h = (omega * normal).sum(axis)
	geom = n_dot_h / (dist_sq + 1e-8)
	diffuse = geom * light_color * albedo / np.pi

	# if nan presents, set value to 0
	diffuse[diffuse != diffuse] = 0.0
	diffuse = th.clamp(diffuse, 0.0, 1.0)

	# clamp the value to [0,1]
	if diffuse_only_flag:
		rendering = diffuse
	else:
		specular = geom * brdf(n_dot_h, rough ** 2, f0) * light_color
		specular[specular != specular] = 0.0
		specular = th.clamp(specular, 0.0, 1.0)
		rendering = diffuse + specular

	rendering = th.clamp(rendering, 1e-10, 1.0) ** (1/2.2)
	return rendering


def GGX(cos_h, alpha):
	c2 = cos_h**2
	a2 = alpha**2
	den = c2 * a2 + (1 - c2)
	return a2 / (np.pi * den**2 + 1e-6)

def Fresnel(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def Fresnel_S(cos, specular):
	sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
	return specular + (1.0 - specular) * sphg

def Smith(n_dot_v, n_dot_l, alpha):
	def _G1(cos, k):
		return cos / (cos * (1.0 - k) + k)
	k = (alpha * 0.5).clamp(min=1e-6)
	return _G1(n_dot_v, k) * _G1(n_dot_l, k)

def norm(vec):
	vec = vec.div(vec.norm(2.0, 0, keepdim=True))
	return vec

def getDir(pos, tex_pos):
	vec = pos - tex_pos
	return norm(vec), (vec**2).sum(0, keepdim=True)

def AdotB(a, b):
	return (a*b).sum(0, keepdim=True).clamp(min=0).expand(3,-1,-1)

def getTexPos(res, size, device):
	x = th.arange(res, dtype=th.float32, device=device)
	x = ((x + 0.5) / res - 0.5) * size

	# surface positions
	y, x = th.meshgrid((x, x))
	z = th.zeros_like(x)
	pos = th.stack((x, -y, z), 0)
	return pos

def render_cpx(normal, albedo, rough, metallic, light_color, f0, size, camera, light_pos, diffuse_only_flag=False):
	# Remove alpha channel for normal and albedo
	if len(normal.shape) == 3 and normal.shape[0] == 4:
		normal = normal[:3,:,:]

	if len(albedo.shape) == 3 and albedo.shape[0] == 4:
		albedo = albedo[:3,:,:]

	axis = 0
	n = normal.shape[1]
	device = normal.device

	# assume albedo in linear space
	albedo = albedo ** 2.2
	
	# update albedo using metallic
	f0 = f0 + metallic * (albedo - f0)
	albedo = albedo * (1.0 - metallic) 

	# n points between [-size/2, size/2]
	tex_pos = getTexPos(n, size, device)
	v, _ = getDir(camera, tex_pos)
	l, dist_l_sq = getDir(light_pos, tex_pos)
	h = norm(l + v)
	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)
	geom = n_dot_l / (dist_l_sq + 1e-8)
	D = GGX(n_dot_h, rough**2)
	F = Fresnel(v_dot_h, f0)
	G = Smith(n_dot_v, n_dot_l, rough**2)
	f1 = albedo / np.pi
	f2 = D * F * G / (4 * n_dot_v * n_dot_l + 1e-6)
	f = f1 + f2
	img = th.clamp(f * geom * light_color, 1e-10, 1.0) ** (1/2.2)

	return img

if __name__ == '__main__':
	pass

