import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IncrementalPCAonGPU():
    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = device
        
        if n_components:
            self.n_components_ = n_components

        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

    def _validate_data(self, X, dtype=torch.float32, copy=True):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype).to(self.device)
        if X.device == torch.device("cpu"):
            X = X.to(self.device)
        if copy:
            X = X.clone()
        return X

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_mean is None:
            last_mean = torch.zeros(X.shape[1], device=X.device)
        if last_variance is None:
            last_variance = torch.zeros(X.shape[1], device=X.device)

        new_sample_count = X.shape[0]
        new_mean = torch.mean(X, dim=0)
        new_sum_square = torch.sum((X - new_mean) ** 2, dim=0)
        
        updated_sample_count = last_sample_count + new_sample_count
        
        updated_mean = (last_sample_count * last_mean + new_sample_count * new_mean) / updated_sample_count
        updated_variance = (last_variance * (last_sample_count + new_sample_count * last_mean ** 2) + new_sum_square + new_sample_count * new_mean ** 2) / updated_sample_count - updated_mean ** 2
        
        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v
    
    def fit(self, X, check_input=True):
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size
        from tqdm import tqdm
        for start in tqdm(range(0, n_samples, self.batch_size_), 'Fitting PCA model'):
            end = min(start + self.batch_size_, n_samples)
            X_batch = X[start:end]
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        if first_pass:
            self.components_ = None
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, torch.tensor([self.n_samples_seen_], device=X.device)
        )

        if self.n_samples_seen_ == 0:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                torch.tensor((self.n_samples_seen_ / n_total_samples.item()) * n_samples, device=X.device)
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)

            if self.singular_values_ is not None and self.components_ is not None:
                X = torch.vstack(
                    (
                        self.singular_values_.view((-1, 1)) * self.components_,
                        X,
                        mean_correction,
                    )
                )

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples.item() - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples.item())

        self.n_samples_seen_ = n_total_samples.item()
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean().item()
        else:
            self.noise_variance_ = 0.0
        return self

    def transform(self, X, check_input=True):
        if check_input:
            X = self._validate_data(X)
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Model must be fitted before transforming data. Please call 'fit' method first or call 'fit_transform' method instead.")
        X = X.to(self.mean_.device)
        X -= self.mean_
        return torch.mm(X, self.components_.T)
    
    def fit_transform(self, X, check_input=True):
        self.fit(X, check_input=check_input)
        return self.transform(X)