import torch
import torchvision
import matplotlib.pyplot as plt

# Choose variants here:
trained_model = False
arch = 'ResNet18'
dataset_name = 'CIFAR10'

## System setup:
import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders(dataset_name, defs,
                                                                      data_path='../data')

models = []
for i in range(2):
    model = torchvision.models.resnet18(pretrained=trained_model)
    model.to(**setup)
    model.eval();
    models.append(model)

dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu());
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu());

# Reconstruct
### Build the input (ground-truth) gradient

idx = 4
# 4 # the frog

img, label = validloader.dataset[idx]
labels = torch.as_tensor((label,), device=setup['device'])
ground_truth = img.to(**setup).unsqueeze(0)
plot(ground_truth);
print([trainloader.dataset.classes[l] for l in labels]);

ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
torchvision.utils.save_image(ground_truth_denormalized, f'{idx}_{arch}_{dataset_name}_input.png')

model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
print(f'Full gradient norm is {full_norm:e}.')

def reconstruct_image():
    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=2,
                  max_iterations=10,
                  total_variation=1e-1,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    rec_machine = inversefed.GradientReconstructor(models, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))

    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output, ground_truth)

    plot(output)
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
              f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");

    data = inversefed.metrics.activation_errors(model, output, ground_truth)

    fig, axes = plt.subplots(2, 3, sharey=False, figsize=(14, 8))
    axes[0, 0].semilogy(list(data['se'].values())[:-3])
    axes[0, 0].set_title('SE')
    axes[0, 1].semilogy(list(data['mse'].values())[:-3])
    axes[0, 1].set_title('MSE')
    axes[0, 2].plot(list(data['sim'].values())[:-3])
    axes[0, 2].set_title('Similarity')

    convs = [val for key, val in data['mse'].items() if 'conv' in key]
    axes[1, 0].semilogy(convs)
    axes[1, 0].set_title('MSE - conv layers')
    convs = [val for key, val in data['mse'].items() if 'conv1' in key]
    axes[1, 1].semilogy(convs)
    convs = [val for key, val in data['mse'].items() if 'conv2' in key]
    axes[1, 1].semilogy(convs)
    axes[1, 1].set_title('MSE - conv1 vs conv2 layers')
    bns = [val for key, val in data['mse'].items() if 'bn' in key]
    axes[1, 2].plot(bns)
    axes[1, 2].set_title('MSE - bn layers')
    fig.suptitle('Error between layers');
    plt.savefig('Figures.png')
    return output

# Reconstruct the image
output = reconstruct_image()
output_denormalized = torch.clamp(output * ds + dm, 0, 1)
torchvision.utils.save_image(output_denormalized, f'{idx}_{arch}_{dataset_name}_output.png')