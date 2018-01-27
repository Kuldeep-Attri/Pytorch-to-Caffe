import caffe
from caffe import layers as L, params as P, to_proto

import torch
import torchvision


import pickle
from functools import partial
from collections import OrderedDict
import argparse
from termcolor import colored

class ModifiedSqueezeNetModel(torch.nn.Module): # This class create an object 
    def __init__(self):
        super(ModifiedSqueezeNetModel, self).__init__()
        model = models.squeezenet1_1(pretrained=True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = True

        final_conv = nn.Conv2d(512,4,kernel_size=1)
        self.classifier = nn.Sequential( 
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13,stride=1))


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0),-1)





def get_me_my_prototxt(model):

	n = caffe.NetSpec()
	n.data = L.Input(input_param={'shape':{'dim':[1,3,224,224]}})

	for i_ in range(len(m.features)): # This will handle the Features but we also need to handle classifier

		if isinstance(m.features[i_], torch.nn.modules.conv.Conv2d): # This is to handle First Convolution Layer
			if i_==0:	
				n.conv1 = L.Convolution(n.data, num_output=m.features[i_].out_channels, kernel_size=m.features[i_].kernel_size[0],
										 stride=m.features[i_].stride[0], pad=m.features[i_].padding[0]) 

		elif isinstance(m.features[i_], torch.nn.modules.activation.ReLU): # This is to handle Relu Layer		
			if i_==1:
				n.relu1 = L.ReLU(n.conv1, in_place=True)

		elif isinstance(m.features[i_], torch.nn.modules.pooling.MaxPool2d): # This is to handle MaxPooling Layer		
			if i_==2:
				n.pool1 = L.Pooling(n.conv1, kernel_size=m.features[i_].kernel_size, stride=m.features[i_].stride, pool=P.Pooling.MAX)	
			if i_==5:
				n.pool2 = L.Pooling(n.fire2_concat, kernel_size=m.features[i_].kernel_size, stride=m.features[i_].stride, pool=P.Pooling.MAX)
			if i_==8:
				n.pool3 = L.Pooling(n.fire4_concat, kernel_size=m.features[i_].kernel_size, stride=m.features[i_].stride, pool=P.Pooling.MAX)
		
		
		elif isinstance(m.features[i_], torchvision.models.squeezenet.Fire): # This is to handle Fire module of SqueezeNet
			
			if i_==3:
				n.fire1_squeeze = L.Convolution(n.pool1, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire1_squeeze_relu = L.ReLU(n.fire1_squeeze, in_place=True)

				n.fire1_expand1x1 = L.Convolution(n.fire1_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire1_expand1x1_relu = L.ReLU(n.fire1_expand1x1, in_place=True)

				n.fire1_expand3x3 = L.Convolution(n.fire1_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire1_expand3x3_relu = L.ReLU(n.fire1_expand3x3, in_place=True)

				n.fire1_concat = L.Concat(n.fire1_expand1x1, n.fire1_expand3x3) 

			if i_==4:
				
				n.fire2_squeeze = L.Convolution(n.fire1_concat, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire2_squeeze_relu = L.ReLU(n.fire2_squeeze, in_place=True)

				n.fire2_expand1x1 = L.Convolution(n.fire2_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire2_expand1x1_relu = L.ReLU(n.fire2_expand1x1, in_place=True)

				n.fire2_expand3x3 = L.Convolution(n.fire2_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire2_expand3x3_relu = L.ReLU(n.fire2_expand3x3, in_place=True)

				n.fire2_concat = L.Concat(n.fire2_expand1x1, n.fire2_expand3x3) 

			if i_==6:	

				n.fire3_squeeze = L.Convolution(n.pool2, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire3_squeeze_relu = L.ReLU(n.fire3_squeeze, in_place=True)

				n.fire3_expand1x1 = L.Convolution(n.fire3_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire3_expand1x1_relu = L.ReLU(n.fire3_expand1x1, in_place=True)

				n.fire3_expand3x3 = L.Convolution(n.fire3_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire3_expand3x3_relu = L.ReLU(n.fire3_expand3x3, in_place=True)

				n.fire3_concat = L.Concat(n.fire3_expand1x1, n.fire3_expand3x3)

			if i_==7:
			
				n.fire4_squeeze = L.Convolution(n.fire3_concat, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire4_squeeze_relu = L.ReLU(n.fire4_squeeze, in_place=True)

				n.fire4_expand1x1 = L.Convolution(n.fire4_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire4_expand1x1_relu = L.ReLU(n.fire4_expand1x1, in_place=True)

				n.fire4_expand3x3 = L.Convolution(n.fire4_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire4_expand3x3_relu = L.ReLU(n.fire4_expand3x3, in_place=True)

				n.fire4_concat = L.Concat(n.fire4_expand1x1, n.fire4_expand3x3)

			if i_==9:

				n.fire5_squeeze = L.Convolution(n.pool3, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire5_squeeze_relu = L.ReLU(n.fire5_squeeze, in_place=True)

				n.fire5_expand1x1 = L.Convolution(n.fire5_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire5_expand1x1_relu = L.ReLU(n.fire5_expand1x1, in_place=True)

				n.fire5_expand3x3 = L.Convolution(n.fire5_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire5_expand3x3_relu = L.ReLU(n.fire5_expand3x3, in_place=True)

				n.fire5_concat = L.Concat(n.fire5_expand1x1, n.fire5_expand3x3)

			if i_==10:

				n.fire6_squeeze = L.Convolution(n.fire5_concat, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire6_squeeze_relu = L.ReLU(n.fire6_squeeze, in_place=True)

				n.fire6_expand1x1 = L.Convolution(n.fire6_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire6_expand1x1_relu = L.ReLU(n.fire6_expand1x1, in_place=True)

				n.fire6_expand3x3 = L.Convolution(n.fire6_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire6_expand3x3_relu = L.ReLU(n.fire6_expand3x3, in_place=True)

				n.fire6_concat = L.Concat(n.fire6_expand1x1, n.fire6_expand3x3)

			if i_==11:

				n.fire7_squeeze = L.Convolution(n.fire6_concat, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire7_squeeze_relu = L.ReLU(n.fire7_squeeze, in_place=True)

				n.fire7_expand1x1 = L.Convolution(n.fire7_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire7_expand1x1_relu = L.ReLU(n.fire7_expand1x1, in_place=True)

				n.fire7_expand3x3 = L.Convolution(n.fire7_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire7_expand3x3_relu = L.ReLU(n.fire7_expand3x3, in_place=True)

				n.fire7_concat = L.Concat(n.fire7_expand1x1, n.fire7_expand3x3)

			if i_==12:	

				n.fire8_squeeze = L.Convolution(n.fire7_concat, num_output=m.features[i_].squeeze.out_channels, 
											kernel_size=m.features[i_].squeeze.kernel_size[0],
											stride=m.features[i_].squeeze.stride[0], pad=m.features[i_].squeeze.padding[0], 
											group=m.features[i_].squeeze.groups)

				n.fire8_squeeze_relu = L.ReLU(n.fire8_squeeze, in_place=True)

				n.fire8_expand1x1 = L.Convolution(n.fire8_squeeze, num_output=m.features[i_].expand1x1.out_channels, 
											kernel_size=m.features[i_].expand1x1.kernel_size[0],
											stride=m.features[i_].expand1x1.stride[0], pad=m.features[i_].expand1x1.padding[0], 
											group=m.features[i_].expand1x1.groups)

				n.fire8_expand1x1_relu = L.ReLU(n.fire8_expand1x1, in_place=True)

				n.fire8_expand3x3 = L.Convolution(n.fire8_squeeze, num_output=m.features[i_].expand3x3.out_channels, 
											kernel_size=m.features[i_].expand3x3.kernel_size[0],
											stride=m.features[i_].expand3x3.stride[0], pad=m.features[i_].expand3x3.padding[0], 
											group=m.features[i_].expand3x3.groups)

				n.fire8_expand3x3_relu = L.ReLU(n.fire8_expand3x3, in_place=True)

				n.fire8_concat = L.Concat(n.fire8_expand1x1, n.fire8_expand3x3)

	for i_ in range(len(m.classifier)):

		if isinstance(m.classifier[i_], torch.nn.modules.dropout.Dropout):
			
			n.drop1 = L.Dropout(n.fire8_concat, dropout_param=dict(dropout_ratio=0.5))
		
		elif isinstance(m.classifier[i_], torch.nn.modules.conv.Conv2d):

			n.conv2 = L.Convolution(n.fire8_concat, num_output=m.classifier[i_].out_channels, kernel_size=m.classifier[i_].kernel_size[0],
										 stride=m.classifier[i_].stride[0], pad=m.classifier[i_].padding[0])
		
		elif isinstance(m.classifier[i_], torch.nn.modules.activation.ReLU):

			n.relu2 = L.ReLU(n.conv2, in_place=True)

		elif isinstance(m.classifier[i_], torch.nn.modules.pooling.AvgPool2d ):	

			n.pool_final = L.Pooling(n.conv2, kernel_size=m.classifier[i_].kernel_size, stride=m.classifier[i_].stride, pool=P.Pooling.AVE)


	n.prob = L.Softmax(n.pool_final)

	return n.to_proto()

def get_me_my_caffemodel(protofile, model):
	
	net = caffe.Net(protofile, caffe.TEST)
	params = net.params

	for name, weights_ in model.state_dict().items():

		if name[0]=='f': # This will handle the feature layers
			key_list = name.split('.')

			count_handle_max_pool=0 # This handle MaxPool layer for Layer # 5 and Layer # 8

			if len(key_list)==3: # Handling the first conv layer
				if name=='features.0.weight':
					params['conv1'][0].data[...] = weights_.numpy()
				elif name=='features.0.bias':
					params['conv1'][1].data[...] = weights_.numpy()	
			
			else:
				if int(key_list[1])>=5 and int(key_list[1])<=8:
					count_handle_max_pool=1
				if int(key_list[1])>=9:
					count_handle_max_pool=2	
				key_name= 'fire'+str(int(key_list[1])-2- count_handle_max_pool )+'_'+key_list[2]
				
				if key_list[3]=='weight':
					params[key_name][0].data[...]=weights_.numpy()

				if key_list[3]=='bias':
					params[key_name][1].data[...]=weights_.numpy()	

		if name[0]=='c': # This will handle the classifier layer
			if name=='classifier.1.weight':
				params['conv2'][0].data[...] = weights_.numpy()
			elif name=='classifier.1.bias':
				params['conv2'][1].data[...] = weights_.numpy()	

	model_name=pt_name.split('.')[0]+'.caffemodel'			
	net.save(model_name)
	print("Congratulations, you converted Pytorch to Caffe...")
	print("Done")


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type = str, default = "model")
	parser.set_defaults(model=False)
	args = parser.parse_args()
	return args




if __name__ == '__main__':

	args = get_args()
	if args.model==False:
		print(colored("Please input model argument...!!!!","red"))

	pickle.load= partial(pickle.load, encoding="latin1")
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

	m = torch.load(args.model, map_location=lambda storage, loc: storage, pickle_module=pickle)
	
	pt_name= args.model+'_squeezenet_p2c_deploy.prototxt'
	with open(pt_name,'w') as f:	
		f.write(str(get_me_my_prototxt(m)))

	get_me_my_caffemodel(pt_name, m)	