from openai import OpenAI
import json
import re
import os
import argparse
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# -----------------------------
# Local FX table (used by tool)
# -----------------------------
FX = {
  2000: {"USD": 0.62, "EUR": 0.57, "GBP": 0.88},
  2001: {"USD": 1.22, "EUR": 1.09, "GBP": 1.77},
  2002: {"USD": 1.50, "EUR": 1.42, "GBP": 2.25},
  2003: {"USD": 1.50, "EUR": 1.70, "GBP": 2.46},
  2004: {"USD": 1.42, "EUR": 1.77, "GBP": 2.61},
  2005: {"USD": 1.34, "EUR": 1.67, "GBP": 2.44},
  2006: {"USD": 1.43, "EUR": 1.81, "GBP": 2.65},
  2007: {"USD": 1.30, "EUR": 1.77, "GBP": 2.62},
  2008: {"USD": 1.30, "EUR": 1.90, "GBP": 2.40},
  2009: {"USD": 1.55, "EUR": 2.15, "GBP": 2.41},
  2010: {"USD": 1.50, "EUR": 2.00, "GBP": 2.35},
  2011: {"USD": 1.60, "EUR": 2.30, "GBP": 2.55},
}

TOOLS = [
  {
    "type": "function",
    "name": "convert_currency",
    "description": "Convert a monetary expression (USD/EUR/GBP) into Turkish Lira (TL) using fixed historical yearly rates (2000-2011). Return TL strings formatted in Turkish style.",
    "parameters": {
      "type": "object",
      "properties": {
        "year": {"type": "integer", "minimum": 2000, "maximum": 2011},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        "amount_text": {"type": "string"}
      },
      "required": ["year", "currency", "amount_text"],
      "additionalProperties": False
    },
    "strict": True
  }
]

def format_tr_lira(n: Decimal) -> str:
    # round to integer TL
    n_int = n.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    s = f"{int(n_int):,}".replace(",", ".")
    return f"{s} TL"

def parse_amount_text(amount_text: str) -> Decimal:
    t = amount_text.strip().lower()

    # If range like "10-12m" or "10 to 12m", take the first number (simple policy)
    t = re.split(r"\s*(?:–|-|to)\s*", t)[0]

    multiplier = Decimal("1")
    if "billion" in t or "bn" in t:
        multiplier = Decimal("1000000000")
    elif "million" in t or re.search(r"\bm\b", t) or t.endswith("m"):
        multiplier = Decimal("1000000")
    elif "thousand" in t or re.search(r"\bk\b", t) or t.endswith("k"):
        multiplier = Decimal("1000")

    # Extract first number (supports 11.4 or 11,4)
    m = re.search(r"(\d+(?:[.,]\d+)?)", t)
    if not m:
        raise ValueError(f"Cannot parse amount_text: {amount_text}")

    num_str = m.group(1).replace(",", ".")
    return Decimal(num_str) * multiplier

def convert_currency_local(year: int, currency: str, amount_text: str) -> dict:
    year = int(year)
    rate = Decimal(str(FX[year][currency]))
    base = parse_amount_text(amount_text)
    tl = base * rate
    return {"tl_formatted": format_tr_lira(tl)}

SYSTEM_PROMPT = """You are an expert Turkish Financial Localization LLM.

Your goal is to take an English financial text and produce a natural, fluent, professional Turkish output that reads like a Turkish financial report / KAP disclosure / institutional research note, while preserving the original meaning, facts, and especially the market sentiment and stance (do not make positive sound negative or vice versa).

When you receive an input, follow this procedure:

1) Year detection
- Detect whether the input mentions a year between 2000 and 2011 (inclusive).
- If a year is found, use that year for all conversions.
- If no year is mentioned, use 2010 by default.

2) Monetary expressions and conversion (TOOL ONLY)
- Detect all monetary amounts and their currencies.
- Supported currencies: USD, EUR, GBP (including symbols and names in English like "$", "€", "£", "US dollars", "euros", "pounds", etc.).
- For every detected monetary expression, call the convert_currency tool with:
  - year: the selected year (2000-2011; default 2010 if not present)
  - currency: one of USD/EUR/GBP
  - amount_text: the raw amount text copied from the input (preserve magnitude words like “million/billion” and abbreviations like “m/bn” if present)
- Replace the original monetary expression with the TL-formatted string returned by the tool.
- Formatting rules for TL:
  - Use Turkish number formatting: thousands separator “.” and decimal separator “,”.
  - Use “TL” suffix (e.g., “1.234.567 TL”, “1.234.567,89 TL”).
  - Keep decimals only when the source has meaningful decimals; otherwise round to the nearest TL.
- Final output must contain only TL. It must NOT contain any foreign currency codes/symbols/names (USD/EUR/GBP/$/€/£/dollar/euro/pound etc.).

3) Turkish language and institutional tone
- Write fluent, idiomatic Turkish.
- Use formal institutional finance tone (KAP/banka raporu).
- Use standard finance phrasing where appropriate (“güçlü görünüm”, “marj baskısı”, “sipariş bakiyesi”, “nakit yaratımı”, “operasyonel kârlılık”, “görünüm”, “beklentiler”).
- Do not add commentary not implied by the source.

4) Turkey-first analog mapping with letter-based variety (MANDATORY)
Rewrite the narrative as if events occurred in Turkey by mapping foreign entities to realistic Turkish analogs.
- Do NOT invent fictional Turkish companies/institutions/places/projects.
- Convert every non Turkish companies/institutions/places/projects to its equvilent context aware Turkish one.
- Use known / verifiable Turkish entities.

5) Sentiment & stance fidelity (STRICT)
- Preserve the market sentiment and stance: positive remains positive, cautious remains cautious, negative remains negative.
- Do not amplify or soften the tone beyond what the source implies.
- Do not introduce new risks/opportunities not present in the input.

Few-shot examples (finance-style Turkish equivalents)

Example 1 (Steel / contract)
Input: “A Finnish steel producer won a bridge steelworks contract worth $11.4 million in 2008.”
Output: “Türkiye’de çelik sektöründe faaliyet gösteren Kardemir, 2008 yılında bir köprü projesine ilişkin çelik üstyapı tedariki ve montajını kapsayan 14.8 milyon TL tutarında sözleşme sağladığını duyurdu.”

Example 2 (Telecom / state stake)
Input: “The minister emphasized that the sale of the state’s stake in a telecom group must be coordinated.”
Output: “Yetkili makamlar, telekomünikasyon grubunda kamu payına ilişkin satış sürecinin ilgili kurumlarla koordinasyon içinde yürütülmesinin kritik önem taşıdığını vurguladı.”

Example 3 (Banking / profitability trend)
Input: “The bank’s net profit rose from €60m to €82m in the first nine months of 2010, supporting a more optimistic outlook.”
Output: “Bankanın net kârı 2010 yılının ilk dokuz ayında 120 milyon TL’den 164 milyon TL’ye yükselerek daha olumlu bir görünüme işaret etti.”

Final output requirements
- Return only the final Turkish text.
- Do not explain rules, steps, or calculations.
- Do not show intermediate math.
- Do not output alternative lists or bracketed candidates.
- Do not mention tools.
- Do not include any foreign currency.
"""

def _to_input_item(obj):
    # Ensure objects returned by SDK are serializable as inputs for next Responses call
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj

def extract_output_text(resp) -> str:
    # Prefer SDK-provided aggregation
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text.strip()

    # Fallback: collect message->content parts
    parts = []
    if hasattr(resp, "output") and resp.output:
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None) or []
                for part in content:
                    if getattr(part, "type", None) == "output_text":
                        parts.append(getattr(part, "text", "") or "")
    return "".join(parts).strip()

def get_usage_tokens(resp) -> dict:
    """
    Returns dict with input_tokens, output_tokens, reasoning_tokens (if available).
    Note: some models include reasoning within output_tokens; availability varies.
    """
    out = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    if hasattr(resp, "usage") and resp.usage:
        out["input_tokens"] = getattr(resp.usage, "input_tokens", 0) or 0
        out["output_tokens"] = getattr(resp.usage, "output_tokens", 0) or 0
        try:
            # Responses API often exposes output_tokens_details.reasoning_tokens
            out["reasoning_tokens"] = (resp.usage.output_tokens_details.reasoning_tokens or 0)
        except Exception:
            out["reasoning_tokens"] = 0
    return out

def run_one(text: str, model: str = None) -> tuple[str, dict]:
    """
    Tool-loop safe translation.
    Returns: (translated_text, token_info_dict)
    """
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    # Accumulate a single conversation state for Responses API
    items = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    total_in = 0
    total_out = 0
    total_reasoning = 0

    # Pricing (optional): set env vars for your model
    # Example for gpt-5-mini: INPUT_COST_PER_1M=0.25, OUTPUT_COST_PER_1M=2.00
    input_cost_per_1m = float(os.getenv("INPUT_COST_PER_1M", "0.25"))
    output_cost_per_1m = float(os.getenv("OUTPUT_COST_PER_1M", "2.00"))

    while True:
        resp = client.responses.create(
            model=model,
            tools=TOOLS,
            input=items,
        )

        usage = get_usage_tokens(resp)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        total_reasoning += usage["reasoning_tokens"]

        # Append model outputs back into context (so it can continue after tool results)
        if hasattr(resp, "output") and resp.output:
            for o in resp.output:
                items.append(_to_input_item(o))

        # Collect tool calls
        tool_calls = []
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == "convert_currency":
                    tool_calls.append(item)

        # If no tool calls, we should have final text
        if not tool_calls:
            final_text = extract_output_text(resp)
            if not final_text:
                final_text = "[ERROR: No output text received from API]"

            cost = (total_in / 1_000_000) * input_cost_per_1m + (total_out / 1_000_000) * output_cost_per_1m
            token_info = {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "thinking_tokens": total_reasoning,  # may be 0 / may be included in output_tokens
                "total_tokens": total_in + total_out,
                "cost": cost,
            }
            return final_text, token_info

        # Execute tool calls and append tool outputs
        for call in tool_calls:
            args = json.loads(call.arguments)
            tool_out = convert_currency_local(
                year=args["year"],
                currency=args["currency"],
                amount_text=args["amount_text"],
            )
            items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(tool_out, ensure_ascii=False),
            })

def process_jsonl(input_file: str, n: int = None, start_index: int = 0):
    """
    Process a JSONL file, translate text fields, and add 'translated_text' key.
    Writes back to the same file.
    
    Args:
        input_file: Path to input JSONL file
        n: Number of elements to translate (None means translate all from start_index)
        start_index: Starting row index (0-based) to begin translation from
    """
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    total_items = len(items)
    
    # Validate start_index
    if start_index < 0:
        start_index = 0
    if start_index >= total_items:
        print(f"Error: Start index {start_index} is beyond file length ({total_items})")
        return
    
    # Slice items from start_index
    items_to_process_list = items[start_index:]
    
    # Determine how many items to process
    if n is not None and n > 0:
        items_to_process = min(n, len(items_to_process_list))
    else:
        items_to_process = len(items_to_process_list)
    
    items_to_process_list = items_to_process_list[:items_to_process]

    print(f"Starting from row index {start_index}")
    print(f"Processing {items_to_process} out of {total_items - start_index} remaining items (total: {total_items})...")
    print("-" * 80)

    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    total_cost = 0.0

    for i, item in enumerate(items_to_process_list, 1):
        # Display actual row number (0-based index + 1 for 1-based display)
        actual_row = start_index + i
        text_to_translate = None

        # Nested structure: train.sentence
        if 'train' in item and isinstance(item['train'], dict):
            if 'sentence' in item['train'] and isinstance(item['train']['sentence'], str):
                text_to_translate = item['train']['sentence']

        # Common top-level fields
        if text_to_translate is None:
            for field in ['text', 'content', 'input', 'original_text', 'english_text', 'sentence', 'prompt']:
                if field in item and isinstance(item[field], str):
                    text_to_translate = item[field]
                    break

        # Any string field fallback
        if text_to_translate is None:
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    text_to_translate = value
                    break
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and len(sub_value) > 10:
                            text_to_translate = sub_value
                            break
                    if text_to_translate:
                        break

        if text_to_translate:
            print(f"\n[Row {actual_row} | {i}/{items_to_process}] Translating...", flush=True)
            try:
                translated, token_info = run_one(text_to_translate)
                item['translated_text'] = translated

                print("\n" + "="*80)
                print("TRANSLATED TEXT:")
                print("="*80)
                print(translated)
                print("="*80 + "\n")

                print("Token Usage:")
                print(f"  Input tokens:    {token_info['input_tokens']:,}")
                print(f"  Output tokens:   {token_info['output_tokens']:,}")
                thinking_display = f"{token_info['thinking_tokens']:,}"
                if token_info['thinking_tokens'] == 0:
                    thinking_display += " (not available / may be included in output tokens)"
                print(f"  Thinking tokens: {thinking_display}")
                print(f"  Total tokens:    {token_info['total_tokens']:,}")
                print(f"  Cost:            ${token_info['cost']:.6f}")

                total_input_tokens += token_info['input_tokens']
                total_output_tokens += token_info['output_tokens']
                total_thinking_tokens += token_info['thinking_tokens']
                total_cost += token_info['cost']

                print("✓ Translated successfully\n", flush=True)
            except Exception as e:
                print(f"  ✗ Error: {e}", flush=True)
                item['translated_text'] = None
        else:
            print(f"\n[Row {actual_row} | {i}/{items_to_process}] ⚠ No text field found to translate", flush=True)
            item['translated_text'] = None

    # Write back to the same file (write all items, including unprocessed ones)
    with open(input_file, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 80)
    print("TRANSLATION SUMMARY")
    print("=" * 80)
    print(f"Items processed:        {items_to_process}")
    print(f"Total input tokens:     {total_input_tokens:,}")
    print(f"Total output tokens:    {total_output_tokens:,}")
    print(f"Total thinking tokens:  {total_thinking_tokens:,}")
    print(f"Total tokens:           {total_input_tokens + total_output_tokens + total_thinking_tokens:,}")
    print(f"Total cost:             ${total_cost:.6f} (${total_cost:.2f})")
    print(f"\n✓ Translation complete! Output saved to {input_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Translate financial text from English to Turkish in JSONL format (tool-safe loop)'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=None,
        help='Number of elements to translate (default: translate all from start index)'
    )
    parser.add_argument(
        '-s',
        '--start',
        type=int,
        default=0,
        help='Starting row index (0-based) to begin translation from (default: 0)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    process_jsonl(args.input_file, args.number, args.start)

if __name__ == '__main__':
    main()
