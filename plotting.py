



val_dataset = ChromosomeDataset(data_dir='../../Desktop/', window=128, length=128, chr=1, itype='Outward')
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=True)
model = newmodel.ConvTransModel(True,128)
untrain_model = newmodel.ConvTransModel(True, 128)

checkpoint=torch.load('../../Desktop/models/epoch=20-step=36540.ckpt',map_location=torch.device('cpu'))
model_weights = checkpoint['state_dict']
model_weights = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
untrain_weights = {k.replace('untrain.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('untrain.')}
    # Edit keys
model.load_state_dict(model_weights)
untrain_model.load_state_dict(untrain_weights)

model.eval()
output=[]
target=[]
untrain_model.eval()
untrain_output=[]
with torch.no_grad():
     for i in tqdm(val_loader):
        inputs, targets = i
        inputs = inputs.transpose(1, 2).contiguous()
        inputs, targets = inputs.float(), targets.float()
        outputs = model(inputs)
        untrain_outputs = untrain_model(inputs)
        output.append(outputs.cpu().view(-1).numpy())
        target.append(targets.cpu().view(-1).numpy())
        untrain_output.append(untrain_outputs.cpu().view(-1).numpy())

np.savetxt('computed_outputs.csv', np.concatenate(output), delimiter=",")
np.savetxt('targets.csv', np.concatenate(target), delimiter=",")
np.savetxt('untrain_outputs.csv', np.concatenate(untrain_output), delimiter=",")