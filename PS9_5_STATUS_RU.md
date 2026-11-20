# PYTHIA PS9.5 — Reversible Transform Discovery Engine

## Что добавлено
Экспериментальный Python-модуль `benchmarks/ps9_5_discovery_engine.py` автоматически ранжирует reversible transform-кандидаты:
- identity;
- modular byte delta на лаге 1,2,3,4,8,16,32;
- XOR predictor на тех же лагах;
- lane-transposition для блоков 4,8,16,32,64;
- bit-plane packing (8 битовых плоскостей).

Кандидаты оцениваются Shannon entropy proxy по transformed bytes + небольшой descriptor overhead. Это предварительный фильтр, **не оценка итогового размера архива**. Все transforms имеют inverse; bit-plane учитывает исходную длину для обратного распаковывания padding.

## Проверки
- `python -m unittest discover -s benchmarks -p 'test_ps9_5_discovery.py'`: 2 теста прошли.
- Round-trip для пустого, короткого, не кратного блоку и длинного случайного буфера.
- Детерминированность списка кандидатов.
- Смешанный тестовый файл 10 MiB: диагностика запускалась на первых 1 MiB. Лучший score у identity/transpose, delta/XOR хуже. Одинаковый entropy proxy у identity и transpose ожидаем, так как перестановка байтов не меняет гистограмму.

## Ограничения
- Это отдельный research module, не включён в production archive selector.
- Entropy не учитывает LZ77, контекстный кодер, framing, checksum и реальный header. Ранжирование может не совпадать с размером архивного потока.
- Rust compile/tests не запускались; данный этап Python reference.

## Следующий технический шаг
Интегрировать candidate discovery с реальным backend cost: сжатие residual каждым доступным энтропийным/LZ backend, плюс descriptor bytes, затем выбирать по полной serialized cost. Это исключит ложные выигрыши от entropy-only proxy.
