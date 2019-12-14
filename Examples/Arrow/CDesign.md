<h2>Introduction</h2>

I would like to talk about some of the design principles underlying the <a href="https://github.com/WinVector/data_algebra"><code>data_algebra</code> package</a> (and also in its sibling <a href="https://github.com/WinVector/rquery"><code>rquery</code> package</a>).

The <a href="https://github.com/WinVector/data_algebra"><code>data_algebra</code> package</a> is a query generator that can act on either <a href="https://pandas.pydata.org"><code>Pandas</code></a> data frames or on <a href="https://en.wikipedia.org/wiki/SQL"><code>SQL</code></a> tables. This is discussed on the <a href="https://github.com/WinVector/data_algebra">project</a> site and the <a href="https://github.com/WinVector/data_algebra/tree/master/Examples">examples directory</a>.  In this note we will set up some technical terminology that will allow us to discuss some of the underlying design decisions.  These are things that when they are done well, the user doesn't have to think much about. Discussing such design decisions at length can obscure some of their charm, but we would like to point out some features here.

<!--more--><p/>

<h2>Background</h2>

We will introduce a few ideas before trying to synthesize our thesis.

<h3>The <code>data_algebra</code></h3>
    
An introduction to the <code>data_algebra</code> package can be found [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Introduction/data_algebra_Introduction.md).  In this note we will discuss some of the inspirations for the package: Codd's relational algebra, experience working with [<code>dplyr</code>](https://CRAN.R-project.org/package=dplyr) at scale, sklearn Pipeline, and category theory.

<h3>sklearn Pipeline</h3>

A major influence on the <code>data_algebra</code> design is <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline"><code>sklearn.pipeline.Pipeline</code></a>. <code>sklearn.pipeline.Pipeline</code> itself presumably become public with <a href="https://github.com/scikit-learn/scikit-learn/commit/b99c76550e3cbe8b57b1ea27b6eb88817a36cb53">Edouard Duchesnay's Jul 27, 2010 commit: "add pipeline"</a>.

<code>sklearn.pipeline.Pipeline</code> maintains a list of steps to be applied to data.  What is interesting is the steps are <em>not</em> functions. Steps are instead objects that implement both a <code>.transform()</code> and a <code>.fit()</code> method.

<code>.transform()</code> typically accepts a data-frame type structure and returns a modified version.  Typical operations include adding a new derived column, selecting columns, selecting rows, and so on.

From a transform-alone point of view the steps compose like functions.  For list <code>[s, t]</code> <code>transform(x)</code> is defined to as:

<pre><code>   transform([s, t], x) := 
      t.transform(s.transform(x))</pre></code>
      
(the glyph "<code>:=</code>" denoting "defined as").

The fit-perspective is where things get interesting.  <code>obj.fit(x)</code> changes the internal state of obj based on the value <code>x</code> and returns a reference to <code>obj</code>.  I.e. it learns from the data and stores this result as a side-effect in the object itself.  In <code>sklearn</code> it is common to assume a composite method called <code>.fit_transform()</code> often defined as: 
<code><pre>   obj.fit_transform(x) := obj.fit(x).transform(x)</pre></code>
(though for our own <a href="https://github.com/WinVector/pyvtreat"><code>vtreat</code></a> package, this is deliberately not the case).

Using <code>.fit_transform()</code> we can explain that in a <code>sklearn Pipeline</code> <code>.fit()</code> is naturally thought of as:
<code><pre>   fit([s, t], x]) := 
      t.fit(s.fit_transform(x))</pre></code>

My point is: <code>sklearn.pipeline.Pipeline</code> generalizes function composition to something more powerful: the ability to both fit and to transform. <code>sklearn.pipeline.Pipeline</code> is a natural way to store a sequence of parameterized or data-dependent data transform steps (such as centering, scaling, missing value imputation, and much more).

This gives as a concrete example of where rigid mindset where "function composition is the only form of composition" would not miss design opportunities.

<h3>Fist class citizens</h3>

We are going to try to design our tools to be "<a href="https://en.wikipedia.org/wiki/First-class_citizen">first class citizens</a>" in the sense of Strachey:

<blockquote>
<strong>First and second class objects.</strong> In <code>ALGOL</code>, a real number may appear in an expression or be assigned to a variable, and either of them may appear as an actual parameter in a procedure call. A procedure, on the other hand, may only appear in another procedure call either as the operator (the most common case) or as one of the actual parameters. There are no other expressions involving procedures or whose results are procedures. Thus in a sense procedures in <code>ALGOL</code> are second class citizens—they always have to appear in person and can never be represented by a variable or expression (except in the case of a formal parameter)...
<p/>
<a href="https://en.wikipedia.org/wiki/First-class_citizen">Quoted in Wikipedia.
Christopher Strachey, "Fundamental Concepts in Programming Languages" in Higher-Order and Symbolic Computation 13:11 (2000); though published in 2000, these are notes from lectures Strachey delivered in August, 1967</a>
</blockquote>

What we will draw out: is if our data transform steps are "first class citizens" we should expect to be able to store them in variables, compose them, examine them, and many other steps.  A function that we can only use or even re-use is not giving us as much as we expect from other types.  Or alternately, if functions don't give us everything we want, we may not want to use them as our only type or abstraction of data processing steps.
  

<h3>Composability</h3>

Most people first encounter the mathematical concept of "composability" in terms of functions.  This can give the false impression that to work with composable design principles, one must shoe-horn the object of interest to be functions or some sort of augmented functions.

This Procrustean view loses a lot of design opportunities.

In mathematics composability is directly studied by the field called "<a href="https://en.wikipedia.org/wiki/Category_theory">Category Theory</a>." So it makes sense to see if category theory may have ideas, notations, tools, and results that may be of use.

<h2>Category Theory</h2>

A lot of the benefit of <a href="https://en.wikipedia.org/wiki/Category_theory">category theory</a> is lost if every time we try to apply category theory (or even just use some of the notation conventions) we attempt to explain <em>all</em> of category theory as a first step.  So I will try to resist that urge here.  I will introduce the bits I am going to use here.

Category theory routinely studies what are called "arrows."  When treated abstractly an arrow has two associated objects called the "domain" and "co-domain." The names are meant to be evocative of the "domain" (space of inputs) and "co-domains" (space of outputs) from the theory of functions.

Functions are commonly defined as having:

<ul>
    <li>A domain, which is a set of values the function can be applied to.</li>
    <li>A co-domain, which is a set of values the function evaluations are contained in (or range).</li>
    <li>A <em>fixed</em> composition rule that if <code>domain(g)</code> is contained in <code>co-domain(f)</code> then:
        <code>g . f</code> is defined as the function such that <code>(g . f)(x) = g(f(x))</code> for all values in 
        the domain of f. Mathematical functions are usually thought of as a specialization of <a href="https://en.wikipedia.org/wiki/Binary_relation">binary relation</a>, and considered to be uniquely determined by their
    evaluations (by the <a href="https://en.wikipedia.org/wiki/Axiom_of_extensionality">axiom of extensionality</a>).</li>
</ul>

Packages that use function composition typically collect functions in lists and define operator composition either through lambda-abstraction or through list concatenation (appealing to 

Category theory differs from function theory in that category theory talks about arrows instead of functions. The theory is careful to keep separate the following two concepts: what arrows are and how arrows are composed.

When using arrows to model a system we expect to be able to specify, with some extra degrees of freedom in specifying:

<ul>
    <li>How to choose domains and co-domains, for categories these do not have to be sets.</li>
    <li>How arrows compose.  For arrows <code>a</code> and <code>b</code> with <code>co-domain(b) = domain(a)</code> then: <code>a . b</code> denotes the composition in the category, and is itself a new arrow in the same category.  Composition is not allowed (or defined) when <code>co-domain(b) != domain(a)</code>.</li>
    <li>How arrows <a href="https://ncatlab.org/nlab/show/action"><em>act</em></a> on items. Arrows can have multiple actions, and arrows can act on items that are not elements of their domains.</li> 
</ul>

An action is a mapping from arrows and items to items.  I.e. <code>action(arrow, item) = new_item</code>. For categories the items may or may not be related to the domain and co-domain. Not all categories have actions, but when they do have actions the action must be compatible with arrow composition.

Good general references on category theory, include:

<ul>
    <li>Steve Awodey, <em>Category Theory, 2nd Edition</em>, Oxford University Press; 2010.</li>
    <li>Emily Riehl, <em>Category Theory in Context</em>, Dover, 2016.</li>
    <li>Saunders Mac Lane, <em>Categories for the Working Mathematician, 2nd Edition</em>, Springer, 1978.</li>
</ul>


Functions have a very ready category theory interpretation as arrows.  Given a function <code>f</code> with domain <code>A</code> and co-domain <code>B</code>, we can think of any triple <code>(f, A', B')</code> as an arrow in a category of functions if <code>A' contained in A</code> and <code>B contained in B'</code>. In this formulation we define the arrow composition of <code><code>(f, A', B')</code></code> and <code>(g, C', D')</code> as <code>(f . g, C', B')</code> where <code>f . g</code>is defined to be the function such that for all <code>x</code> in domain <code>x</code> we have:
<code><pre>   (f . g)(x) := f(g(x)) </pre></code>

We will call the application of a function to a value as an example of an "<a href="https://ncatlab.org/nlab/show/action">action</a>." A function <code>f()</code> "acts on its domain" and <code>f(x)</code> is the action of <code>f</code> on <code>x</code>.  For functions we can define the action "<code>apply</code>" as:
<code><pre>   apply(f, x) := f(x)</pre></code>

The extra generalization power we get from moving away from functions to arbitrary arrows (that might not correspond to functions) comes from the following:

<ul>
<li>Arrow composition does <em>not</em> have to be function composition.</li>
<li>Arrows can have multiple actions, and <a href="https://ncatlab.org/nlab/show/action">act</a> on items that are <em>not</em> elements of their domain.</li>
<li>Arrows have a notion of equality, but this notion <em>can differ</em> from having identical actions.</li>
</ul>

To be a category a few conditions must be met, including: the composition must be associative and we must have some identity arrows. By "associative composition" we mean, it must be the case that for arrows <code>a</code>,
<code>b</code>, and <code>c</code> (with appropriate domains and co-domains):
<code><pre>   (a . b) . c = a . (b . c) </pre></code>

Our action must also associate with arrow composition.  That is we must have for values <code>x</code> we must have for co-variant actions:
<code><pre>   apply(a . b, x) = apply(a, apply(b, x))</pre></code>

Or for contra-variant actions:
<code><pre>   apply(a . b, x) = apply(b, apply(a, x))</pre></code>


The idea is: the arrow <code>a . b</code> must have an action equal to the actions of a and b composed as functions. That is: arrow composition and actions can differ from function composition and function application, but they must be at least somewhat similar in that they remain <a href="https://en.wikipedia.org/wiki/Associative_property">associative</a>.

We now have the background to see that category theory arrows differ from functions in that arrows are more general (we can pick more of their properties) and require a bit more explicit bookkeeping.

<h2>Back to <code>sklearn.pipeline.Pipeline</code></h2>

We now have enough notation to attempt a crude category theory description of <code>sklearn.pipeline.Pipeline</code>.

Define our <code>sklearn.pipeline.Pipeline</code> category <code>P</code> as follows:

<ul>
<li>We have only one object called <code>0</code>. All arrows will have domain and co-domain equal to <code>0</code>, i.e.: we are not doing any interesting pre-condition checking in this category. This sort of category is called a "<a href="https://ncatlab.org/nlab/show/monoid#inamonoidalcategory">monoid</a>."</li>
<li>The arrows of our category are lists of steps.  
Steps are again <code>Python</code> objects
that define <code>.transform()</code>, <code>.fit()</code>, and <code>.fit_transform()</code> methods.</li>
<li>Composition <code>a1 . a2</code> is defined as the list concatenation: <code>a2 + a1</code>.  "<code>+</code>" being <code>Python</code>'s list concatenate in this case, and the order set to match <code>sklearn.pipeline.Pipeline</code> list order convention.</li>
<li>We define an action called "<code>transform_action</code>" defined as:

<code><pre>   transform_action([step1, step2, ..., stepk], x) := 
      stepk.transform(... step2.transform(step1.transform(x)) )</pre></code>
</li>
</ul>

To see this is a category (and a category compatible action) we must check associativity of the composition (which in this case is list concatenation) and associativity of the action with respect to list concatenation.  

We can also try to model the <code>.fit_transform()</code> methods.  We will not try to model the side-effect that <code>.fit_transform()</code> changes state of the arrows (to have the fit information in each step).  But we can at least define an action (with side effects) as follows:

<ul>
<li>We define an action called "<code>fit_transform</code>" defined as:
<code><pre>   fit_transform_action([step1, step2, ..., stepk], x) := 
      stepk.fit_transform(... step2.fit_transform(step1.fit_transform(x)) )</pre></code>
</li>
</ul>

To confirm this is an action (ignoring the side-effects), we would want check is if the following equality holds or not:
<code><pre>  fit_transform_action(a . b, x) =
      fit_transform_action(b, fit_transform_action(a, x))
   </pre></code>

The above should follow by brute pushing notation around (assuming we have defined <code>fit_transform_action</code> correctly, and sufficiently characterized <code>.fit_transform()</code>).

Notice we didn't directly define a "<code>fit_action</code>" action, as it isn't obvious that has a not obvious that has a nice associative realization. This an opportunity for theory to drive design; notation considerations hint that <code>fit_transform()</code> may be more fundamental than, and thus preferred over, <code>fit()</code>.

The category theory concepts didn't so-much design <code>sklearn.pipeline.Pipeline</code>, but give us a set of criteria to evaluate <code>sklearn.pipeline.Pipeline</code> design.  We trust the category theory point of view is useful as it emphasizes associativity (which is a great propriety to have), and is routinely found to be a set of choices that work in complicated systems.  The feeling being: the design points category theory seems to suggest, turn out to be the ones you want down the round.

<h2>The <code>data_algebra</code></h2>

Now that we have some terminology, lets get back to the <code>data_algebra</code>

<h3>What is the <code>data_algebra</code>?</h3>

<ul>
<li>
    <a href="https://github.com/WinVector/data_algebra"><code>data_algebra</code></a> is a package for building up complex data manipulation queries
 <code>data_algebra</code> queries are first class citizens in the Strachey sense (can be: passed as an argument, returned from a function, modified, assigned to a variable, printed, inspected, and traversed as a data structure).
</li>
<li>
 The operators are essentially those of the Codd relational algebra (select rows/columns, join, union-all, extend, project, and window functions).
</li>
<li>
    Composition is left to right using <a href="https://en.wikipedia.org/wiki/Method_chaining">method chaining</a>.
</li>
<li>
    Queries can be realized by transforming to <code>SQL</code> (targeting <code>PostgeSQL</code>, <code>Spark</code>, and other implementations), or as acting on <code>Pandas</code> data (we are hoping to extend this to <code>modin</code>, <code>RAPIDS</code>, and others).
    </li>
    <li>The <code>data_algebra</code> has an <a href="https://www.r-project.org"><code>R</code></a> sibling package group
        (<a href="https://github.com/WinVector/rquery"><code>rquery</code></a>/<a href="https://github.com/WinVector/rqdatatable"><code>rqdatatable</code></a>) similar to <a href="https://CRAN.R-project.org/package=dplyr"><code>dplyr</code></a>.</li>
</ul>

An introduction to the <code>data_algebra</code> can be found <a href="https://github.com/WinVector/data_algebra">here</a>.

We now have the terminology to concisely state a <code>data_algebra</code> design principle: use general concepts (such as category theory notation) to try and ensure <code>data_algebra</code> transforms are first class citizens (i.e. we can do a lot with them and to them).

<h3>The naive functional view</h3>

If we were to again take a mere functional view of the <a href="https://github.com/WinVector/data_algebra"><code>data_algebra</code></a> we would say the <code>data_algebra</code> is a set of functions that operate on data.  They translate data frames to new data frames using <a href="https://en.wikipedia.org/wiki/Relational_algebra">Codd</a>-inspired operations. We could think of the <code>data_algebra</code> as acting on data on the right, and acting on <code>data_algebra</code> operators on the left.

However, this is not the right abstraction.  <code>data_algebra</code> methods primarily map data transforms to data transforms. However even this is a "too functional view". It makes sense to think of <code>data_algebra</code> operators as arrows, and the whole point of arrows is composition.

<h3>The categorical view</h3>

The <code>data_algebra</code> can be mapped to a nice category.  The idea being something that can be easily mapped to an orderly system, is it self likely an somewhat orderly system.

Good references on the application of category theory to concrete systems (including databases) include:

<ul>
    <li>David I. Spivak, <em>Category Theory for the Sciences</em>; The MIT Press, 2014.</li>
    <li>Brendan Fong, David I. Spivak, <em>An Invitation to Applied Category Theory: Seven Sketches in Compositionality</em>; Cambridge University Press, 2019.</li>
</ul>


Our <code>data_algebra</code> category <code>D</code> is defined as follows.

<ul>
    <li>The objects of our category are single table <a href="https://en.wikipedia.org/wiki/Database_schema">schemas</a>.  By "single table schema" mean mean only the list of column names (and optionally column types) for a set of named tables.  We are not modeling invariants, or cross-table relations.</li>
    <li>The arrows of our category are <code>data_algebra</code> operator chains.</li>
    <li>Composition of arrows in our category is a very general query composition.  We will demonstrate query composition in a bit, but as a hint it is not function composition or list concatination.</li>
</ul>

Some notes on the category theory interpretation of the <code>data_algebra</code> package can be found <a href="https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md">here</a>.

Let's demonstrate the above with <code>Python</code> code.  The <code>data_algebra</code> allows for the specification of data transforms as first class objects.

First we import some modules and create some example data.


```python
from data_algebra.data_ops import *
import pandas

d = pandas.DataFrame({
    'x': [1, 2, 3],
    'y': [3, 4, 4],
})

d
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



To specify adding a new derived column <code>z</code> we would write code such as the following.


```python
td = describe_table(d)

a = td.extend(
    { 'z': 'x.mean()' },
    partition_by=['y']
)

a
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x.mean()'},
       partition_by=['y'])



We can let this transform act on data.


```python
a.transform(d)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



We can compose this transform with more operations to create a composite transform.


```python
b = a.extend({
    'ratio': 'y / x'
})

b
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x.mean()'},
       partition_by=['y']) .\
       extend({
        'ratio': 'y / x'})



As a bonus we can also map the above transform to a <a href="https://en.wikipedia.org/wiki/SQL"><code>SQL</code></a> query representing the same action in databases.


```python
from data_algebra.SQLite import SQLiteModel

print(
    b.to_sql(db_model=SQLiteModel(), pretty=True)
)
```

    SELECT "x",
           "y",
           "z",
           "y" / "x" AS "ratio"
    FROM
      (SELECT "x",
              "y",
              avg("x") OVER (PARTITION BY "y") AS "z"
       FROM ("data_frame") "SQ_0") "SQ_1"


All of this is the convenient interface we expect users will want.  However, if we asked that all operators specified their expected input schema (or their domain) we have the category <code>D</code>.  We don't expect users to do such, but we have code supporting this style of notation to show that the <code>data_algebra</code> is in fact related to a nice category over schemas.

Lets re-write the above queries as formal category arrows.


```python
from data_algebra.arrow import *

a1 = DataOpArrow(a)

print(str(a1))
```

    [
     'data_frame':
      [ x: <class 'numpy.int64'>, y: <class 'numpy.int64'> ]
       ->
      [ x, y, z ]
    ]
    


The above is rendering the arrow as just its domain and co-domain. The domain and co-domains are just single-table schemas: lists of column names (possibly with column types).

We can get a more detailed representation of the arrow as follows.


```python
print(a1.__repr__())
```

    DataOpArrow(
     TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x.mean()'},
       partition_by=['y']),
     free_table_key='data_frame')


Or we can examine the domain and co-domain directly.  Here we are using a common category theory trick: associating the object with the identity arrow of the object.  So what we are showing as domain and co-domains are actually identity arrows instead of objects.


```python
a1.dom()
```




    DataOpArrow(
     TableDescription(
     table_name='',
     column_names=[
       'x', 'y']),
     free_table_key='')




```python
a1.cod()
```




    DataOpArrow(
     TableDescription(
     table_name='',
     column_names=[
       'x', 'y', 'z']),
     free_table_key='')



Now we can write our second transform step as an arrow as follows.


```python
a2 = DataOpArrow(a1.cod_as_table().extend({
    'ratio': 'y / x'
}))

a2
```




    DataOpArrow(
     TableDescription(
     table_name='',
     column_names=[
       'x', 'y', 'z']) .\
       extend({
        'ratio': 'y / x'}),
     free_table_key='')



We took extra steps, that most users will not want to take, to wrap the second-stage (<code>a2</code>) operations as an arrow.  Being an arrow means that we have a domain and co-domain that can be used to check if operations are composable.

A typical user would not work with arrow, but instead work with the data algebra which itself is a shorthand for the arrows. That is: the users may want the power of a category, but they don't want to be the one handling the extra bookkeeping. To add an extra operation a user would work directly with <code>a</code> and just write the following.


```python
a.extend({
    'ratio': 'y / x'
})
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x.mean()'},
       partition_by=['y']) .\
       extend({
        'ratio': 'y / x'})



The above has substantial pre-condition checking and optimizations (as it is merely user facing shorthand for the arrows).

The more cumbersome arrow notation (that requires the specification of pre-conditions) has a payoff: managed arrow composition. That is: complex operator pipelines can be directly combined.  We are not limited to extending one operation at a time.

If the co-domain of arrow matches the domain of another arrow we can compose them left to right as follows.


```python
a1.cod() == a2.dom()
```




    True




```python
composite = a1 >> a2

composite
```




    DataOpArrow(
     TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x.mean()'},
       partition_by=['y']) .\
       extend({
        'ratio': 'y / x'}),
     free_table_key='data_frame')



And when this isn't the case, composition is not allowed.  This is exactly what we want as this means the preconditions (exactly which columns are present) for the second arrow are not supplied by the first arrow.


```python
a2.cod() == a1.dom()
```




    False




```python
try:
    a2 >> a1
except ValueError as e:
    print("Caught: " + str(e))
```

    Caught: extra incoming columns: {'z', 'ratio'}


An important point is: for this arrow notation composition is not mere list concatenation or function composition.  Here is an example that makes this clear.


```python
b1 = DataOpArrow(TableDescription(column_names=['x', 'y'], table_name=None). \
   extend({
    'x': 'x + 1',
    'y': 7
}))

b1
```




    DataOpArrow(
     TableDescription(
     table_name='',
     column_names=[
       'x', 'y']) .\
       extend({
        'x': 'x + 1',
        'y': '7'}),
     free_table_key='')




```python
b2 = DataOpArrow(TableDescription(column_names=['x', 'y'], table_name=None). \
   extend({
    'y': 9
}))
```

Now watch what happens when we use "<code>>></code>" to compose the arrow <code>b1</code> and <code>b2</code>.


```python
b1 >> b2
```




    DataOpArrow(
     TableDescription(
     table_name='',
     column_names=[
       'x', 'y']) .\
       extend({
        'x': 'x + 1',
        'y': '9'}),
     free_table_key='')



Notice in this special case the composition of <code>b1</code> and <code>b2</code> is a single extend node combining the operations and eliminating the dead-value <code>7</code>.  The idea is: the package has some freedom to define composition as long as it is associative.  In this case we have an optimization at the compose step so the composition is not list concatenation or function composition.

As we have said, a typical user will not take the time to establish pre-conditions on steps.  So they are not so much working with arrows but with operators that can be specialized to arrows.  An actual user might build up the above pipeline as follows.


```python

TableDescription(column_names=['x', 'y'], table_name=None). \
   extend({
    'x': 'x + 1',
    'y': 7
    }). \
   extend({
    'y': 9
    })
    
```




    TableDescription(
     table_name='',
     column_names=[
       'x', 'y']) .\
       extend({
        'x': 'x + 1',
        'y': '9'})



We <a href="http://www.win-vector.com/blog/2019/12/what-is-new-for-rquery-december-2019/">recently</a> demonstrated this sort of optimization in the <code>R</code> <code>rquery</code> package.

In the above example the user still benefits from the category theory design. As they composed left to right the system was able to add in the pre-conditions for them.  The user only needs to set pre-conditions for non-trivial right-hand side pipelines.

<h2>Conclusion</h2>

The advantage the <code>data_algebra</code> package gets from category theory is: it lets us design the package action (how the package works on data) somewhat independently from operator composition. This gives us a lot more design room and power than a strict function composition or list concatenation theory would give us.


