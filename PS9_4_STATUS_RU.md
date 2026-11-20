# PYTHIA PS9.4 — Algebraic Search (experimental)

## Реализовано в исходниках
- Новый standalone модуль `src/algebraic_search.rs` для причинного нелинейного предсказания над байтами modulo 256.
- 8 термов: линейные x[i-1], x[i-2], квадрат, произведение, XOR, AND, OR и куб.
- Каноническая сериализация параметров (число термов + пары term/coefficient); проверка диапазона, сортировки, дубликатов и длины.
- Lossless forward/inverse; inverse использует только уже восстановленную историю.
- Детерминированный multi-start coordinate optimizer с histogram Renyi-2 proxy. Его результат экспериментальный и не является доказанно оптимальным.
- Unit tests: обратимость, параметры, детерминизм синтеза.
- Независимый Python reference и диагностический entropy test.

## Важно
PS9.4 пока не интегрирован в production archive pipeline и не имеет ModelId/descriptor registration. Не использовать его для создания архивов, пока не выполнена интеграция с exact archive-cost evaluator и тесты совместимости.

Rust compiler/cargo в окружении не обнаружены; Rust unit tests не запускались. Выполнены Python syntax check и reference roundtrip.

## Диагностика
На искусственном 65,536-byte последовательном образце с выбранным вручную набором термов Shannon entropy histogram proxy изменился с 6.5850 до 5.4942 бит/байт. Это диагностическая статистика, не сравнение реального архивного размера и не доказательство выигрыша на универсальных файлах.
