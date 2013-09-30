function X0=quantize_data(X0)

nBins=256;

if ~isa(X0,'uint8')
  xMin = min(X0) -.01;
  xMax = max(X0) +.01;
  xStep = (xMax-xMin) / (nBins-1);
  X0 = uint8(bsxfun(@times,bsxfun(@minus,X0,xMin),1./xStep));
end


end