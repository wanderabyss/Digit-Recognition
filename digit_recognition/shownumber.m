function [h, imageShow] = shownumber(X)
% Gray Image
 colormap(gray);

 % Figure out how many images came in
 amount = size(X, 1);

 % Make it so that it comes out as a square
 t = sqrt(amount);
 t = t - mod(t, 1);

 imageShow = zeros(30*t,30*t);

 count = 1;
 level = 0;
 level2 = 0;
 for n=1:amount
 for i=(2+level2):(29+level2)
 for j=(2+level):(29+level)
 imageShow(i,j) = X(n,count) + 100;
 count = count+1;
 end
 end
 count = 1;
 level = level+30;
 if (mod(n,t) == 0)
 level2 = level2+30;
 level = 0;
 end
 end
 % Display Image
 h = imagesc(imageShow);
 % Hide the axis
 axis image off

 drawnow;
 end