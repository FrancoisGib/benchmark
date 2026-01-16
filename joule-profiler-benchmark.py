import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import subprocess
    import tempfile
    from pathlib import Path
    import polars as pl
    import os
    import altair as alt
    import pyarrow
    from tqdm import tqdm
    import time
    import nbformat
    return Path, alt, mo, os, pl, subprocess, tempfile, time, tqdm


@app.function
def fibo(n):
    # don't want two one
    yield 0
    a, b = 1, 1
    for _ in range(n):
        yield b
        a, b = b, a + b


@app.function
def yield_every(step: int, n: int):
    for i in range(0, n * step + 1, step):
        yield i


@app.function
def hz_to_s(hz):
    return 0 if not hz or hz <= 0 else 1 / hz


@app.cell
def _(os):
    JOULE_PROFILER = "./joule-profiler/target/debug/joule-profiler"
    SUDO_PASSWORD = os.getenv("SUDO_PASSWORD")

    DEFAULT_NB_ITERATIONS = 100
    DEFAULT_NB_POLLING = 30
    DEFAULT_SLEEP_TIME = 0.01
    DEFAULT_N_BODY_VALUE = 50000
    DEFAULT_POLLING_FUNCTION = yield_every(1000, DEFAULT_NB_POLLING)
    return (
        DEFAULT_NB_ITERATIONS,
        DEFAULT_N_BODY_VALUE,
        DEFAULT_POLLING_FUNCTION,
        DEFAULT_SLEEP_TIME,
        JOULE_PROFILER,
    )


@app.cell
def _(JOULE_PROFILER, Path, pl, subprocess, tempfile):
    def run_joule_profiler(command, rapl_polling=None, extra_args=None):
        extra_args = extra_args or []
        if rapl_polling == 0:
            rapl_polling = None

        # if SUDO_PASSWORD == None:
        #     raise Exception("SUDO_PASSWORD must be set")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"

            cmd = [
                "sudo",
                "-S",
                JOULE_PROFILER,
                "simple",
                "--csv",
                "--jouleit-file",
                str(csv_path),
            ]

            polling_s = hz_to_s(rapl_polling)
            if rapl_polling is not None:
                cmd += ["--rapl-polling", str(polling_s)]

            cmd += extra_args
            cmd += ["--", *command]

            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            return pl.read_csv(csv_path, separator=";")
    return (run_joule_profiler,)


@app.cell
def _(
    DEFAULT_NB_ITERATIONS,
    DEFAULT_N_BODY_VALUE,
    DEFAULT_POLLING_FUNCTION,
    DEFAULT_SLEEP_TIME,
    pl,
    run_joule_profiler,
    time,
    tqdm,
):
    def generate_iters(nb_iterations=DEFAULT_NB_ITERATIONS, sleep_time=DEFAULT_SLEEP_TIME, polling_function=DEFAULT_POLLING_FUNCTION, nbody_value=DEFAULT_N_BODY_VALUE):
        dfs = []

        for polling in tqdm(polling_function):
            iter_dfs = []
            for _ in range(nb_iterations):
                res = run_joule_profiler(
                    command=["python3", "nbody.py", str(nbody_value)],
                    rapl_polling=polling
                )
                iter_dfs.append(res)
                time.sleep(sleep_time)

            df_polling = pl.concat(iter_dfs)
            df_polling = df_polling.with_columns(pl.lit(polling).alias("expected_frequency"))
            df_polling = df_polling.with_columns((pl.col("measure_delta") / 1000).alias("measure_delta"))
            dfs.append(df_polling)

        return pl.concat(dfs)
    return (generate_iters,)


@app.cell
def _(mo):
    mo.md(r"""
    Group dataframe rows by expected_frequency.
    """)
    return


@app.cell
def _(pl):
    def df_group_by_expected_frequency(df):
        return df.group_by("expected_frequency").agg([pl.mean(c) for c in df.columns if c !="expected_frequency"])
    return (df_group_by_expected_frequency,)


@app.cell
def _(mo):
    mo.md(r"""
    Remove the unrelevant columns in our dataset.
    """)
    return


@app.function
def df_removed_unused_columns(df):
    return df.drop(["command", "exit_code"])


@app.cell
def _(mo):
    mo.md(r"""
    Aggregate the dataset:
    - Select only the CORE_0 RAPL domain, the expected frequency, the number of measures and the duration to compute the real frequency
    - Compute the real_frequency for each polling rate
    - Compute the energy of the CORE_0 domain
    - Remove measure_count and duration_ms columns (not needed anymore)
    - Put real_frequency column in first
    """)
    return


@app.cell
def _(pl):
    _MS_TO_SECONDS_RATIO = 1000
    _MICRO_JOULE_TO_JOULE_RATIO = 1000000

    def df_aggregate(df):
        df = df.select(["expected_frequency", "CORE_0", "measure_count", "duration_ms", "measure_delta"])

        df = df.with_columns([
            (1000 / pl.col("measure_delta")).alias("real_frequency"),

            (pl.col("CORE_0") / _MICRO_JOULE_TO_JOULE_RATIO).alias("core_energy")
        ])

        df = df.with_columns([
            pl.when(pl.col("expected_frequency") == 0)
              .then(0)
              .otherwise(pl.col("real_frequency"))
              .alias("real_frequency")
        ])
        df = df.drop(["measure_count", "duration_ms"])
        df = df.select(["real_frequency", *[c for c in df.columns if c != "real_frequency"]])

        return df
    return (df_aggregate,)


@app.cell
def _(alt):
    def generate_energy_chart(df, regression: bool = True):
        df_plot = df

        _min_energy = df_plot["core_energy"].min()
        _max_energy = df_plot["core_energy"].max()
        _margin = (_max_energy - _min_energy) * 0.05
        _min_energy -= _margin
        _max_energy += _margin

        energy_chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X("expected_frequency", title="Polling frequency (Hz)"),
            y=alt.Y("core_energy", title="Energy consumption (J)", scale=alt.Scale(domain=[_min_energy, _max_energy])),
            tooltip=["expected_frequency", "core_energy"]
        )

        _regression_line = energy_chart.transform_regression(
            "expected_frequency", "core_energy", method="linear"
        ).mark_line(color="red", strokeDash=[5,5])

        if regression:
            return (energy_chart + _regression_line).interactive()
        else:
            return energy_chart.interactive()
    return (generate_energy_chart,)


@app.cell
def _(alt, pl):
    def generate_frequency_chart(df, identity_line: bool = True):
        df_plot = df

        min_val = min(df_plot["expected_frequency"].min(),
                      df_plot["real_frequency"].min())
        max_val = max(df_plot["expected_frequency"].max(),
                      df_plot["real_frequency"].max())

        frequency_chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X(
                "expected_frequency",
                title="Expected polling frequency (Hz)",
                scale=alt.Scale(domain=[min_val, max_val])
            ),
            y=alt.Y(
                "real_frequency",
                title="Real polling frequency (Hz)",
                scale=alt.Scale(domain=[min_val, max_val])
            ),
            tooltip=["expected_frequency", "real_frequency"]
        )

        if identity_line:
            identity_df = pl.DataFrame({
                "x": [0, df_plot["expected_frequency"].max()],
                "y": [0, df_plot["expected_frequency"].max()],
            })

            identity = alt.Chart(identity_df).mark_line(
                color="red",
                strokeDash=[5, 5]
            ).encode(
                x="x",
                y="y"
            )

            return (frequency_chart + identity).interactive()

        return frequency_chart.interactive()
    return (generate_frequency_chart,)


@app.cell(disabled=True)
def _(generate_iters):
    _nb_polling = 20
    _nb_iterations = 10
    _polling_function = yield_every(1000, _nb_polling)

    high_frequency_iters = generate_iters(nb_iterations=_nb_iterations, polling_function=_polling_function)
    return (high_frequency_iters,)


@app.cell
def _(df_group_by_expected_frequency, high_frequency_iters, mo):
    _high_frequency_df_before_groups = df_removed_unused_columns(high_frequency_iters)
    high_frequency_df = df_group_by_expected_frequency(_high_frequency_df_before_groups)

    mo.vstack([_high_frequency_df_before_groups, high_frequency_df])
    return (high_frequency_df,)


@app.cell
def _(df_aggregate, high_frequency_df):
    high_frequency_df_agg = df_aggregate(high_frequency_df)
    high_frequency_df_agg
    return (high_frequency_df_agg,)


@app.cell
def _(generate_energy_chart, generate_frequency_chart, high_frequency_df_agg):
    high_frequency_chart = generate_frequency_chart(high_frequency_df_agg)
    high_frequency_energy_chart = generate_energy_chart(high_frequency_df_agg)
    return high_frequency_chart, high_frequency_energy_chart


@app.cell
def _(high_frequency_chart, high_frequency_energy_chart, mo):
    mo.vstack([high_frequency_chart, high_frequency_energy_chart])
    return


@app.cell
def _(generate_iters):
    _nb_polling = 100
    # _nb_iterations = 100
    _nb_iterations = 2
    _nbody_value = 500
    _polling_function = yield_every(10, _nb_polling)

    low_frequency_df_iters = generate_iters(nb_iterations=_nb_iterations, polling_function=_polling_function, nbody_value=_nbody_value)
    low_frequency_df_iters = df_removed_unused_columns(low_frequency_df_iters)
    low_frequency_df_iters.write_csv("data.csv")
    low_frequency_df_iters
    return (low_frequency_df_iters,)


@app.cell(disabled=True)
def _(df_group_by_expected_frequency, low_frequency_df_iters, mo):
    _low_frequency_df_before_groups = df_removed_unused_columns(low_frequency_df_iters)
    low_frequency_df = df_group_by_expected_frequency(_low_frequency_df_before_groups)

    mo.vstack([_low_frequency_df_before_groups, low_frequency_df])
    return (low_frequency_df,)


@app.cell
def _(df_aggregate, low_frequency_df):
    low_frequency_df_agg = df_aggregate(low_frequency_df)
    low_frequency_df_agg
    return (low_frequency_df_agg,)


@app.cell
def _(
    generate_energy_chart,
    generate_frequency_chart,
    low_frequency_df_agg,
    mo,
):
    low_frequency_chart = generate_frequency_chart(low_frequency_df_agg)
    low_frequency_energy_chart = generate_energy_chart(low_frequency_df_agg)

    mo.vstack([low_frequency_chart,low_frequency_energy_chart])
    return (low_frequency_chart,)


@app.cell
def _(low_frequency_chart):
    low_frequency_chart
    return


if __name__ == "__main__":
    app.run()
