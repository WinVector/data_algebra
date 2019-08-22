

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



```python
import data_algebra.pending_eval

def recursive_example_ex(n, d=1):
    if n <= 1:
        return d
    else:
        # eliminate tail-call by using exception
        # instead of return recursive_example_ex(n-1, d+1)
        raise data_algebra.pending_eval.PendingFunctionEvaluation(recursive_example_ex, n - 1, d + 1)

data_algebra.pending_eval.eval_using_exceptions(recursive_example_ex, 100000)
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
            raise data_algebra.pending_eval.PendingFunctionEvaluation(self.f_, n - 1, d + 1)

    def f(self, n, d=1):
        return data_algebra.pending_eval.eval_using_exceptions(self.f_, n=n, d=d)

o = C()
o.f(100000)


```




    100000


