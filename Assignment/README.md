# cv_course_hinton
This repository stores the codes and assignemnts completed within the cv course. 

## The 2th Week Assignment
[Requirements]
1. To code a median filtering function instead of using cv2.medianBlur API in Python. To try to implement the function with a faster method than bacis method.
2. To write a pseudo code for RANSAC algorithm.

[Results]
1. The time consumings of two different median blur algorithms can be obtained from the output. Some experiment results are shown below.

algorithm|radius=1|radius=2|radius=3|radius=5|radius=9|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Classic|7.18s |32.33s |101.88s |542.70s | 4791.78s|
|Huang|7.01s | 7.59s| 8.39s| 9.79s|11.85s |

[References] 
1. S. Perreault, P. Hĺębert, "Median filtering in constant time", IEEE Trans. Image Process., vol. 16, pp. 2389-2394, 2007. 
2. https://www.cnblogs.com/yoyo-sincerely/p/6058944.html.
