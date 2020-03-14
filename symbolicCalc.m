clear
clc

syms c11 c12 c13 c21 c22 c23 c31 c32 c33;

C_3d = [c11, c12, c13;
    c21, c22, c23;
    c31, c32, c33];

I1 = trace(C_3d);
I2 = 1/2*(trace(C_3d)*trace(C_3d) - trace(C_3d*C_3d));
I3 = det(C_3d);

dI3 = [diff(I3,c11), diff(I3,c12), diff(I3,c13);
    diff(I3,c21), diff(I3,c22), diff(I3,c23);
    diff(I3,c31), diff(I3,c32), diff(I3,c33)];

dI2 = [diff(I2,c11), diff(I2,c12), diff(I2,c13);
    diff(I2,c21), diff(I2,c22), diff(I2,c23);
    diff(I2,c31), diff(I2,c32), diff(I2,c33)];

dI1 = [diff(I1,c11), diff(I1,c12), diff(I1,c13);
    diff(I1,c21), diff(I1,c22), diff(I1,c23);
    diff(I1,c31), diff(I1,c32), diff(I1,c33)];


A_1 = [cos(pi/12), -sin(pi/12), 0];

M_1 = zeros(3,3);

for x=1:3
    for y=1:3
        M_1(x,y) = A_1(x)*A_1(y);
    end
end

J4_1 = trace(C_3d*M_1);

dJ4_1 = [diff(J4_1,c11), diff(J4_1,c12), diff(J4_1,c13);
    diff(J4_1,c21), diff(J4_1,c22), diff(J4_1,c23);
    diff(J4_1,c31), diff(J4_1,c32), diff(J4_1,c33)];

%% 2D
C_2d = [c11, c12;
    c21, c22;];

I1 = trace(C_2d);
I2 = 1/2*(trace(C_2d)*trace(C_2d) - trace(C_2d*C_2d));
I3 = det(C_2d);

dI3 = [diff(I3,c11), diff(I3,c12);
    diff(I3,c21), diff(I3,c22)];

dI2 = [diff(I2,c11), diff(I2,c12);
    diff(I2,c21), diff(I2,c22)];

dI1 = [diff(I1,c11), diff(I1,c12);
    diff(I1,c21), diff(I1,c22)];
%% 2D tissue
A_1 = [sqrt(0.5), sqrt(0.5)];

M_1 = zeros(2,2);

for x=1:2
    for y=1:2
        M_1(x,y) = A_1(x)*A_1(y);
    end
end

J4_1 = trace(C_2d*M_1);

dJ4_1 = [diff(J4_1,c11), diff(J4_1,c12);
    diff(J4_1,c21), diff(J4_1,c22)];
