function [] = printDememsion(net)

for i=1:numel(net.blob_names)
    name = net.blob_names{i};
    fprintf('%s: %d x %d x %d x %d\n', name, net.blobs(name).shape);
end
end

