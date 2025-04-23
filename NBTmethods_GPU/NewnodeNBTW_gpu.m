function newMat = NewnodeNBTW_gpu(M, a, n, t)
% testinv = shortcircleinv(M,n,t,a);
  E = {};
  e = ones(n,n);
 for i = 1:t
     E{i} = e;
 end
 E = blkdiag(E{:});

if ~isa(a, 'sym')
    E = gpuArray(E);
else
    E = sym(E);
    
    
end
try
    scaled_M = a*M;  %a*M is forbidden for symbolic a
catch
    scaled_M = a*gather(M);
end

scaled_transpose = trans(scaled_M,n,t);


Z = E - newcircle(scaled_transpose,scaled_M,n,t); % Z = E - newcircle(scaled_transpose,a*M,n,t)

testinv =circleinv_gpu(Z,n,t);

 
X = full(newcircle(scaled_M,testinv,n,t));

if ~isa(a,'sym')
    identity = eye(n*t, 'gpuArray');
else
    identity = sym(eye(n*t));
end


test = eye(n*t) - X + dd_gpu(scaled_M*X,n,t);



% Y = test \ eye(n*t);
% newMat = Y;

newMat = test \ ones(n*t,1);
newMat = newMat(1:n);

end