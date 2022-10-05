% Given: 
%       - x_r,s1: sensor 1 wrt the robot coordinate frame
%       - x_s1,s2: sensor 2 in coordinate frame of sensor 1
%       - x_wr1: Pose of the robot at t1 wrt world frame
%               - mu_wr1, sigma_1 of the above state
%       - x_wr1: Pose of the robot at t2 wrt world frame
%               - mu_wr2, sigma_2 of the above state
%       - sigma_12: cross covariance between the robot's pose at t1 and t2

% SCT-Q1 ==================================================
% Desired: pose of sensor s1 wrt sensor s2
% function: ssc_inverse
% Given:
%       - sensor 2 in the frame of sensor 1
% Returns:
%       - sensor 1 in the frame of sensor 2
%       - Jacobian of the transformation
[X_primeQ1,JminusQ1] = ssc_inverse(x_s1s2)
% =========================================================


% SCT-Q2 ===================================================
% Desired: Mean pose of sensor s1 in world frame at t1
% function: ssc_head2tail
% Given:
%       - mean pose of robot wrt the world frame at time t1
%       - position of sensor 1 wrt to robot frame
% Returns:
%       - sensor 1 in the world frame at time t1
%       - Jacobian of the transformation
[X_ws1Q2,JplusQ2] = ssc_head2tail(mu_wr1,x_rs1)

% Get the covariance of s1's pose in world frame at time t1
P_lcQ2 = JplusQ2*[Sigma_1, zeros(6); zeros(6), zeros(6)]*JplusQ2'
% =========================================================



% SCT-Q3 ===================================================
% Desired: Mean pose and Jacobian of sensor s2 in frame w at t1
% function: 
% Given:
%       - sensor 2 in sensor 1 (from first problem)
%       - sensor 1 in robot frame
%       - mean of robot position in world frame
% Returns:
%       - mean pose of sensor 2 in the world frame at time t1
%       - Jacobian of the transformation
[X_ws2Q3,JplusQ3] = ssc_head2tail(mu_wr1,ssc_head2tail(x_rs1, X_primeQ1))

% Get the covariance of s2's pose in world frame at time t1
P_lcQ3 = JplusQ3*[Sigma_1, zeros(6); zeros(6), zeros(6)]*JplusQ3'
% =========================================================


% SCT-Q4 =================================================== 
%       - Plots the 3-sigma confidence ellipses for global robot
%               poses xwr1, xwr2 in the world frame
hold("on")
draw_ellipse(mu_wr1, Sigma_1(1:2, 1:2), 9)
draw_ellipse(mu_wr2, Sigma_2(1:2, 1:2), 9)

%       - Relative robot pose at time t2 wrt time t1 and plot the
%               first-order (x,y)^3 sigma confidence ellipse
[X_r1r2Q4,JQ4] = ssc_tail2tail(mu_wr1, mu_wr2)
figure;
draw_ellipse(X_r1r2Q4, Sigma_12(1:2, 1:2), 9)
% =========================================================


% SCT-Q5 ===================================================

% =========================================================

