mat = zeros(32,32) ; 
p=0;
for i=1:32
    for j=1:32
        p=p+1;
        mat(i,j) =p-1 ; 
       
    end
end

mat

kernel = zeros(8,8) ; 
k=0;
for i=1:8
    for j=1:8
        k=k+1;
        kernel(i,j) =k-1; 
       
    end
   
 
end

kernel

C=conv2(mat,kernel);
