% read output
fid=fopen('optix-homogcube.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[61,61,61,1]);

% visualize
cw=res;
mcxplotvol(log(cw));