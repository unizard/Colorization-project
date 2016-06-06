net = network(1, 3);
net.name = 'Colorization Network';
net.inputConnect = [1 0 0]';
net.layerConnect = [0 0 0; 1 0 0; 0 1 0];
net.biasConnect = [1; 1; 1];
net.outputConnect = [0 0 1];

net.adaptFcn = 'adaptwb';
net.derivFcn = 'defaultderiv';
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainscg';

