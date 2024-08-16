from fastai.torch_core import TensorBase


class Transform(metaclass=_TfmMeta):
    "Delegates (`__call__`,`decode`,`setup`) to (<code>encodes</code>,<code>decodes</code>,<code>setups</code>) if `split_idx` matches"
    split_idx, init_enc, order, train_setup = None, None, 0, None

    def __init__(self, enc=None, dec=None, split_idx=None, order=None):
        self.split_idx = ifnone(split_idx, self.split_idx)
        if order is not None:
            self.order = order
        self.init_enc = enc or dec
        if not self.init_enc:
            return

        self.encodes, self.decodes, self.setups = TypeDispatch(), TypeDispatch(), TypeDispatch()
        if enc:
            self.encodes.add(enc)
            self.order = getattr(enc, 'order', self.order)
            if len(type_hints(enc)) > 0:
                self.input_types = union2tuple(first(type_hints(enc).values()))
            self._name = _get_name(enc)
        if dec:
            self.decodes.add(dec)

    @property
    def name(self):
        return getattr(self, '_name', _get_name(self))

    def __call__(self, x, **kwargs):
        return self._call('encodes', x, **kwargs)

    def decode(self, x, **kwargs):
        return self._call('decodes', x, **kwargs)

    def __repr__(self):
        return f'{self.name}:\nencodes: {self.encodes}decodes: {self.decodes}'

    def setup(self, items=None, train_setup=False):
        train_setup = train_setup if self.train_setup is None else self.train_setup
        return self.setups(getattr(items, 'train', items) if train_setup else items)

    def _call(self, fn, x, split_idx=None, **kwargs):
        if split_idx != self.split_idx and self.split_idx is not None:
            return x
        return self._do_call(getattr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        if not _is_tuple(x):
            if f is None:
                return x
            ret = f.returns(x) if hasattr(f, 'returns') else None
            return retain_type(f(x, **kwargs), x, ret)
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
        return retain_type(res, x)


class InplaceTransform(Transform):
    "A `Transform` that modifies in-place and just returns whatever it's passed"

    def _call(self, fn, x, split_idx=None, **kwargs):
        super()._call(fn, x, split_idx, **kwargs)
        return x


class TabularProc(InplaceTransform):
    "Base class to write a non-lazy tabular processor for dataframes"

    def setup(self, items=None, train_setup=False):  # TODO: properly deal with train_setup
        super().setup(getattr(items, 'train', items), train_setup=False)
        # Procs are called as soon as data is available
        return self(items.items if isinstance(items, Datasets) else items)

    @property
    def name(self):
        return f"{super().name} -- {getattr(self,'__stored_args__',{})}"


def _apply_cats(voc, add, c):
    if not (hasattr(c, 'dtype') and isinstance(c.dtype, CategoricalDtype)):
        return pd.Categorical(c, categories=voc[c.name][add:]).codes + add
    return c.cat.codes + add  # if is_categorical_dtype(c) else c.map(voc[c.name].o2i)


def _decode_cats(voc, c):
    return c.map(dict(enumerate(voc[c.name].items)))


class Categorify(TabularProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 1

    def setups(self, to):
        store_attr(
            classes={n: CategoryMap(to.iloc[:, n].items, add_na=(n in to.cat_names)) for n in to.cat_names}, but='to'
        )

    def encodes(self, to):
        to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))

    def decodes(self, to):
        to.transform(to.cat_names, partial(_decode_cats, self.classes))

    def __getitem__(self, k):
        return self.classes[k]


def flatten_check(inp, targ):
    "Check that `inp` and `targ` have the same number of elements and flatten them."
    inp, targ = TensorBase(inp.contiguous()).view(-1), TensorBase(targ.contiguous()).view(-1)
    test_eq(len(inp), len(targ))
    return inp, targ


def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred, targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()
