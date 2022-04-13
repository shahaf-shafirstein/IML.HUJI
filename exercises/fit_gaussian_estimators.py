from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    rand_samples = np.random.normal(10,1,(1000,))
    y = UnivariateGaussian()
    y.fit(rand_samples)
    print((y.mu_, y.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # Creating a list "distance" that keeps the absolut distance between the
    # estimated expectation and the real expectation (which equals 10). Then
    # shows the results on a graph.
    distance = []
    for i in range(10,1000,10):
        estimator = UnivariateGaussian().fit(rand_samples[:i])
        distance.append(abs(estimator.mu_-10))
    a_range = range(10,1000,10)
    plt.plot(a_range, distance)
    plt.title("Q2 - Sample Mean Is Consistent")
    plt.xlabel("Number Of Samples")
    plt.ylabel("Distance (Estimated - Real Expectation)")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # Calculates the PDF's for the random samples and shows them on a graph.
    pdfs = y.pdf(rand_samples)
    plt.plot(rand_samples,pdfs, 'o')
    plt.xlabel("Samples Values")
    plt.ylabel("PDF")
    plt.title("Q3 - Empirical PDF Of Fitted Model")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = np.array([[1, 0.2, 0, 0.5],[0.2, 2, 0, 0],[0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    mu = np.array([0, 0, 4, 0])
    multi_rand_samples = np.random.multivariate_normal(mu, cov_matrix, size = 1000)
    y = MultivariateGaussian()
    y.fit(multi_rand_samples)
    print(y.mu_)
    print(y.cov_)

    # Question 5 - Likelihood evaluation
    # Calculates different expectation vectors according to the values from
    # linspace and calculates the likelihood respectively. Also extract the
    # maximum likelihood for the next question.
    log_likelihood_values = np.zeros((200,200))
    values = np.linspace(-10, 10, 200)
    for f1 in range(values.size):
        for f3 in range(values.size):
            sec_mu = np.array([values[f1], 0, values[f3], 0])
            log_likelihood_values[f1,f3]=MultivariateGaussian.log_likelihood(sec_mu,
                                                                  cov_matrix,multi_rand_samples)
            if f1==0 and f3==0: # Initialize the values in the first iteration
                max_likelihood = log_likelihood_values[f1,f3]
                f1f3_max_likelihood = (values[f1], values[f3])
                # calculating for the next question.
            if log_likelihood_values[f1,f3] > max_likelihood:
                max_likelihood = log_likelihood_values[f1,f3]
                f1f3_max_likelihood = (values[f1], values[f3])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=values,y=values,z=log_likelihood_values))
    fig.update_layout(title="Q5 - Log Likelihood Evaluation",xaxis_title="f3 "
                                                                     "Values", yaxis_title="f1 Values")
    fig.show()

    # Question 6 - Maximum likelihood
    print(f1f3_max_likelihood)



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    # s = np.array(
    #     [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3,
    #      1, -4, 1, 2, 1,
    #      -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1,
    #      0, 3, 5, 0, -2])
    # c = UnivariateGaussian()
    # c.fit(s)
    # print(UnivariateGaussian.log_likelihood(10,1,s))