

```python
def recursive_example(n, d=1):
    if n <= 1:
        return d
    else:
        return recursive_example(n - 1, d + 1)

try:
    recursive_example(10000)
except Exception as ex:
    print(ex)
    
```

    maximum recursion depth exceeded in comparison


Note: this throw-pattern can *only* be used in the case where tail-calls are only called by tail-calls, as the raise will throw-through intermediate code.


```python
import data_algebra.pending_eval as pe

def recursive_example_ex(n, d=1):
    if n <= 1:
        return d
    else:
        # eliminate tail-call by using exception
        # instead of return recursive_example_ex(n-1, d+1)
        raise pe.PendingFunctionEvaluation(
            recursive_example_ex, n - 1, d + 1)

pe.eval_using_exceptions(recursive_example_ex, 100000)
```




    100000




```python
class C:

    def f_(self, n, d=1):
        if n <= 1:
            return d
        else:
            # Eliminate tail-call by using an exception.
            # instead of: return self.f_(n-1, d+1), use:
            raise pe.PendingFunctionEvaluation(
                self.f_, n - 1, d + 1)

    def f(self, n, d=1):
        return pe.eval_using_exceptions(self.f_, n=n, d=d)

o = C()
o.f(100000)
```




    100000



Or something closer to the original [return-based system](https://www.kylem.net/programming/tailcall.html).


```python
def recursive_example_rt(n, d=1):
    if n <= 1:
        return d
    else:
        return pe.PendingFunctionEvaluation(
            recursive_example_rt, n - 1, d + 1)


pe.eval_using_exceptions(recursive_example_rt, 100000)
```




    100000




```python
class Cr:

    def f_(self, n, d=1):
        if n <= 1:
            return d
        else:
            return pe.PendingFunctionEvaluation(
                self.f_, n - 1, d + 1)

    def f(self, n, d=1):
        return pe.eval_using_exceptions(self.f_, n=n, d=d)

o2 = Cr()
o2.f(100000)
```




    100000


