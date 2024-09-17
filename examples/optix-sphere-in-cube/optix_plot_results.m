% read output
fid=fopen('optix-spheretest.bin');
output=fread(fid,'float64');

% retrieve spatial results
res=reshape(output,[61,61,61,1]);

% convert to cw solution and visualize
cw=res;
mcxplotvol(log(cw));