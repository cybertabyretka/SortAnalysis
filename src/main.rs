#[derive(Default, Clone)]
struct OperationCounter {
    comparisons: usize,
    swaps: usize,
}

impl OperationCounter {
    /// Returns the total number of operations (comparisons + swaps). 
    fn total(&self) -> usize {
        self.comparisons + self.swaps
    }
}

/// A simple bubble sort implementation that counts comparisons and swaps.
fn bubble_sort(arr: &mut [i32], counter: &mut OperationCounter) {
    let n = arr.len();

    for i in 0..n {
        for j in 0..n - 1 - i {
            counter.comparisons += 1;

            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                counter.swaps += 1;
            }
        }
    }
}

#[derive(Clone)]
struct ExperimentPoint {
    n: f64,
    operations: f64,
}

type SortFn = fn(&mut [i32], &mut OperationCounter);

struct InfinityAnalyzer {
    sort_fn: SortFn,
    intercept: f64,
    slope: f64,
}

impl InfinityAnalyzer {
    fn new(sort_fn: SortFn) -> Self {
        Self {
            sort_fn,
            intercept: 0.0,
            slope: 0.0,
        }
    }

    /// Runs the sorting experiments for a range of input sizes and collects the operation counts.
    fn run_experiments(
        &self, start: usize, step: usize, count: usize,
    ) -> Vec<ExperimentPoint> {
        let mut points = Vec::new();
        for i in 0..count {
            let n = start + i * step;
            let mut data: Vec<i32> =
                (0..n).map(|_| rand::random_range(0..100000)).collect();
            let mut counter = OperationCounter::default();
            (self.sort_fn)(&mut data, &mut counter);

            points.push(ExperimentPoint {
                n: n as f64,
                operations: counter.total() as f64,
            });
        }
        points
    }

    /// Builds a new table of points where both n and operations are transformed to their logarithmic values.
    pub fn build_log_table(&self, points: &[ExperimentPoint]) -> Vec<ExperimentPoint> {
        points
            .iter()
            .map(|p| ExperimentPoint {
                n: (p.n).log10(),
                operations: (p.operations).log10(),
            })
            .collect()
    }

    /// Computes the slope and intercept for the log-log linear regression.
    pub fn compute_log_log_lin_reg(
        &mut self, log_points: &[ExperimentPoint]
    ) -> Result<(f64, f64), String> {
        if log_points.is_empty() {
            return Err("Not enough valid points for regression".into());
        }
        let len = log_points.len();
        let n_mean = log_points.iter().map(|p| p.n as f64).sum::<f64>() / len as f64;
        let op_mean = log_points.iter().map(|p| p.operations as f64).sum::<f64>() / len as f64;
        let mut nop_cov = 0f64;
        let mut n_var = 0f64;
        for i in 0..len {
            nop_cov += (log_points[i].n as f64 - n_mean) * (log_points[i].operations as f64 - op_mean);
            n_var += (log_points[i].n as f64 - n_mean).powi(2);
        }
        if n_var == 0.0 {
            return Err("Variance of log(n) is zero".into());
        }
        let slope = nop_cov / n_var;
        let intercept = op_mean - slope * n_mean;
        self.slope = slope;
        self.intercept = intercept;
        Ok((slope, intercept))
    }

    /// Returns the estimated parameters α and C based on the computed slope and intercept.
    fn parameters_alpha_c(&self) -> (f64, f64) {
        (self.slope, 10f64.powf(self.intercept))
    }
}

/// Utility function to print the experiment results in a tabular format.
fn print_table(points: &[ExperimentPoint]) {
    println!("N\tOperations");
    for p in points {
        println!("{}\t{}", p.n, p.operations);
    }
}

fn main() -> Result<(), String> {
    // Create an instance of the InfinityAnalyzer with the bubble sort function.
    let mut analyzer = InfinityAnalyzer::new(bubble_sort);
    let points = analyzer.run_experiments(100, 100, 15);
    print_table(&points);

    // Build the log-log table and compute the linear regression to estimate α and C.
    let log_points = analyzer.build_log_table(&points);
    analyzer.compute_log_log_lin_reg(&log_points)?;
    let (alpha, c) = analyzer.parameters_alpha_c();

    // Print the estimated parameters.
    println!("\nEstimated parameters:");
    println!("C ≈ {}", c);
    println!("α ≈ {}", alpha);
    Ok(())
}