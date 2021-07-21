from train import *

class PixelClassifier_Tester(Rooftop_Dataset):
    def __init__(
        self,
        model = None,
        model_path = "models/model_v5.pth",
        loss_params = {
         "loss_type": "bce",
         "reduction": "sum"
        },
        max_pool = True,
        useGPU = True,
        show_status = True
    ):
        Rooftop_Dataset.__init__(self, isTrain = False)
        self.model = model
        self.model_path = model_path
        if not self.model and self.model_path:
            self.load_model()
        
        self.loss_params = loss_params
        self.loss_criterion = get_loss(self.loss_params["loss_type"], self.loss_params["reduction"])
        self.useGPU = useGPU
        self.show_status = show_status
        
        self.device = torch.device("cpu")
        if self.useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.show_status: INFO("Using GPU for inferencing and testing")
        
        self.max_pool = max_pool
        if self.max_pool: self.maxPool = nn.MaxPool2d((max_pool,max_pool), stride = 1, padding = max_pool//2)
        
    def load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()
        
    def predict(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            if self.max_pool: outputs = self.maxPool(outputs)
        return outputs.cpu()
    
    def evaluate(self, test_loader):
        losses = list()
        for batch_idx, (test_data, test_labels) in tqdm(enumerate(test_loader)):
            predicted_masks = self.predict(test_data.to(self.device))
            loss = self.loss_criterion(predicted_masks, test_labels)
            losses.append(loss.item())
        return np.mean(losses)
    
    def process_image(self, image):
        if isinstance(image, str): image = read_image(image)
        elif isinstance(image, np.array): image = Image.fromarray(image)
        return self.transform_image(image).unsqueeze(dim = 0)
    
    def predict_image(self, image, return_type = "PIL.Image"):
        # input - str, image, np.array, torch.Tensor
        output = self.predict(self.process_image(image))
        output = output.cpu().detach()
        output = output.permute(0,2,3,1).squeeze(dim = 0)
        
        if return_type == "np.array":
            return output.numpy()
        elif return_type == "PIL.Image":
            return Image.fromarray(output)
        else: # torch.Tensor
            return output
        
    def inference_large(self, image_path, patch_size = (500, 500), path = "output"):
        img = read_image(image_path).convert("RGB")
        img_array = np.array(img)
        
        image_transform = transforms.Compose([transforms.ToTensor()])
        
        row_tensors = list()
        
        for xidx in range(0, img_array.shape[0], patch_size[0]):
        
            input_patches = list()
            for yidx in range(0, img_array.shape[1], patch_size[1]):
                crop = img_array[xidx:(xidx + patch_size[0]), yidx:(yidx + patch_size[1])]
                input_patches.append(image_transform(Image.fromarray(crop)))
            
            row_outputs = self.predict(torch.stack(input_patches)).detach()
            row_tensors.append(torch.hstack([r for r in row_outputs.squeeze()])) # (image_size / patch_size)  x patch_size x patch_size
        
        output_mask = (np.round(torch.vstack(row_tensors).numpy()) * 255).astype(np.uint8)
        try:
            save_image(output_mask, path)
        except:
            pass
        return Image.fromarray(output_mask)
        
if __name__=='__main__':
    test_dataset = Rooftop_Dataset(isTrain = False)
    test_loader = get_data_loader(test_dataset)