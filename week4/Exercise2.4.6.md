1. So far we took the rules for derivatives for granted. Using the definition and limits prove the properties for $ f(x) = c $, $ f(x) = x^n $, $ f(x) = e^x $, $ f(x) = log(x) $

Ans: 
$$
\begin{aligned}
f'(x) 
&= \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \\
\\ \text{For }f(x) = c , \\
f'(x) &= \lim_{h \to 0} \frac{c - c}{h} = \lim_{h \to 0} \frac{0}{h} = 0 \\
\\ \text{For }f(x) = x^n, \\
f'(x) 
&= \lim_{h \to 0} \frac{(x+h)^n-x^n}{h} \\
&= \lim_{h \to 0} \frac{\displaystyle \sum_{k=0}^n \binom{n}{k}x^{n-k}h^k - x^n}{h} \\
&= \lim_{h \to 0} \frac{x^n + nx^{n-1}h + \displaystyle \sum_{k=2}^n \binom{n}{k}x^{n-k}h^k - x^n}{h} \\
&= \lim_{h \to 0} (nx^{n-1} + \sum_{k=2}^n \binom{n}{k}x^{n-k}h^{k-1}) \\
&= nx^{n-1}
\\ \text{For }f(x) = e^x, \\
f'(x) 
&= \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} \\
&= \lim_{h \to 0} \frac{e^xe^h - e^x}{h} \\
&= \lim_{h \to 0} \frac{e^x(e^h - 1)}{h} \\
&= e^x\underbrace{\lim_{h \to 0} \frac{e^h - 1}{h}}_{\text{Standard Limit }=1} \\
&= e^x
\\ \text{For }f(x) = log(x), \\
f'(x) 
&= \lim_{h \to 0} \frac{\log(x+h) - \log(x)}{h} \\
&= \lim_{h \to 0} \frac{\log(\frac{x+h}{x})}{h} \\
&= \lim_{h \to 0} \frac{1}{h} \log(1 + \frac{h}{x}) \\
&= \lim_{h \to 0} \log \left( (1 + \frac{h}{x})^{\frac{1}{h}} \right) \\
&= \log \left( \underbrace{\lim_{h \to 0} (1 + \frac{h}{x})^{\frac{x}{h}}}_{e} \right)^{\frac{1}{x}} \\
&= \log(e^{\frac{1}{x}}) \\
&= \frac{1}{x}
\end{aligned}
$$

2. In the same vein, prove the product, sum, and quotient rule from first principles.

Ans:

Sum rule:
$$
\begin{aligned}
\frac{d}{dx}[f(x) + g(x)] 
&= \lim_{h \to 0} \frac{(f(x+h) + g(x+h)) - (f(x) + g(x))}{h} \\
&= \lim_{h \to 0} \frac{(f(x+h) - f(x)) + (g(x+h) - g(x))}{h} \\
&= \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} + \lim_{h \to 0} \frac{g(x+h) - g(x)}{h} \\
&= \frac{d}{dx}f(x) + \frac{d}{dx}g(x)
\end{aligned}
$$

Product rule:
$$
\begin{aligned}
\frac{d}{dx}[f(x)g(x)]
&= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h} \\
&= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x)- f(x)g(x)}{h} \\
&= \underbrace{\lim_{h \to 0} f(x+h)}_{f(x)\text{(by continuity)}}\lim_{h \to 0}\frac{g(x+h) - g(x)}{h} + g(x)\lim_{h \to 0} \frac{f(x+h)- f(x)}{h} \\
&= f(x)\lim_{h \to 0} \frac{g(x+h) - g(x)}{h} + g(x)\lim_{h \to 0} \frac{f(x+h)- f(x)}{h} \\
&= f(x)\frac{d}{dx}g(x) + g(x)\frac{d}{dx}f(x)
\end{aligned}
$$

Quotient Rule:
$$
\begin{aligned}
\frac{d}{dx} \frac{f(x)}{g(x)} 
&= \lim_{h \to 0} \frac{\frac{f(x+h)}{g(x+h)} - \frac{f(x)}{g(x)} }{h} \\
&= \lim_{h \to 0} \frac{\frac{f(x+h)g(x)}{g(x+h)g(x)} - \frac{f(x)g(x+h)}{g(x+h)g(x)} }{h} \\
&= \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{g(x+h)g(x)h} \\
&= \frac{1}{g(x)}\underbrace{\lim_{h \to 0} \frac{1}{g(x+h)}}_{\frac{1}{g(x)}\text{(by continuity)}}\lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{h} \\
&= \frac{1}{g(x)^2}\lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{h} \\
&= \frac{1}{g(x)^2}\lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x) - f(x)g(x+h) + f(x)g(x)}{h} \\
&= \frac{1}{g(x)^2}(\lim_{h \to 0} g(x)\frac{f(x+h) - f(x)}{h} - f(x)\frac{g(x+h) - g(x)}{h}) \\
&= \frac{g(x)\frac{d}{dx}f(x) - f(x)\frac{d}{dx}g(x)}{g^2(x)} \\
\end{aligned}
$$

3. Prove that the constant multiple rule follows as a special case of the product rule.

Ans:

Constant multiple rule:

$$
\begin{aligned}
\\ \text{Let } g(x) &= C \\g'(x) &= 0\\
\frac{d}{dx}[Cf(x)]
&= \frac{d}{dx}[f(x)g(x)] \\
&= f(x)\frac{d}{dx}g(x) + g(x)\frac{d}{dx}f(x) \\
&= f(x) * 0 + g(x)\frac{d}{dx}f(x) \\
&= C\frac{d}{dx}f(x) \\
\end{aligned}
$$

4. Calculate the derivative of $ f(x) = x^x$

Ans:


$$
\begin{aligned}
\text{Let } u &= x\ln{x} ,\\
f(x) 
&= x^x = e^{\ln{x^x}} \\
&= e^{x\ln{x}} = e^u\\
f'(x)
&= \frac{d}{dx}e^{x\ln{x}} \\
&= \frac{d}{du}e^u\frac{d}{dx}x \ln{x} \\
&= e^u(x\frac{d}{dx}\ln{x} + \ln{x}\frac{d}{dx}x) \\
&= e^u(x\frac{1}{x} + \ln{x}*1)\\
&= e^{x\ln{x}}(\ln{x} + 1) \\
&= x^x(\ln{x} + 1)
\end{aligned}
$$

5. What does it mean that $f'(x) = 0 $ for some $x$? Give an example of a function $f$ and a location $x$ for which this might hold.

Ans:

It means that changing x does not increase/decrease the value of $f(x)$, which means it reachs a local optimal or a saddle point. Geometrically, the tangent line at that point has a 0 slope, so it is a horizontal line.

Example, 
$$
\begin{aligned}
f(x) &= x^2 - 3 \\
\text{at } x = 0, \\
f'(x) &= 2x = 0 \\
f(x) &= -3
\end{aligned}
$$

6. Plot the function $y = f(x) = x^3 - \frac{1}{x}$ and plot its tangent line at $x = 1$.

Ans:

![alt text](<CleanShot 2025-11-23 at 22.28.35@2x.png>)

7. Find the gradient of the function $f(x) = 3x_1^2 + 5e^{x_2}$

Ans:

$$
\begin{aligned}
\text{Gradient } 
&= \nabla f(x) \\
&= [\partial x_1f(x), \partial x_2f(x)]^T \\
&= [6x_1, 5e^{x_2}]^T
\end{aligned}
$$

8. What is the gradient of the function $f(x) = \|\mathbf{x}\|_2$? What happens for $\mathbf{x} = 0$?

Ans:

$$
\begin{aligned}
\text{Let } u &= \|\mathbf{x}\|_2^2 = \mathbf{x}^T\mathbf{x} \\
f(x) &= \sqrt{u} = u^{1/2} \\

\text{Gradient } 
&= \nabla f \\
&= \frac{df}{du} \nabla u \\
&= (u^{1/2})' \nabla \|\mathbf{x}\|_2^2 \\
&= \frac{1}{2\sqrt{u}} \nabla \mathbf{x}^T\mathbf{x} \\
&= \frac{1}{2\|\mathbf{x}\|_2} 2\mathbf{x} \\
&= \frac{\mathbf{x}}{\|\mathbf{x}\|} \\

\text{For } x &= 0, \\
\text{Gradient }
&= \frac{0}{0} \text{which is undefined}

\end{aligned}
$$

9. Can you write out the chain rule for the case where $u = f(x, y, z)$ and $x = x(a, b), y = y(a, b),$ and $z = z(a,b)$

Ans:


$$
\begin{aligned}
\frac{\partial u}{\partial a} = \frac{\partial u}{\partial x}\frac{\partial x}{\partial a} + \frac{\partial u}{\partial y}\frac{\partial y}{\partial a} + \frac{\partial u}{\partial z}\frac{\partial z}{\partial a} \\
\frac{\partial u}{\partial b} = \frac{\partial u}{\partial x}\frac{\partial x}{\partial b} + \frac{\partial u}{\partial y}\frac{\partial y}{\partial b} + \frac{\partial u}{\partial z}\frac{\partial z}{\partial b} \\
\end{aligned}
$$

10. Given a function $f(x)$ that is invertible, compute the derivative of its inverse $f^{-1}(x)$. Here we have that $f^{-1}(f(x)) = x$ and conversely $f(f^{-1}(y)) = y$. Hint: use these properties in your derivative.


$$
\begin{aligned}
\text{Let } g &= f^{-1}(x) \\
x &= f(g) \\
\text{Differentiate both sides}&\text{ with respect to $x$.} \\
\frac{d}{dx}x &= \frac{d}{dx}f(g) \\
1 &= f'(g)(f^{-1})'(x) \\
(f^{-1})'(x) &= \frac{1}{f'(g)}

\end{aligned}
$$
